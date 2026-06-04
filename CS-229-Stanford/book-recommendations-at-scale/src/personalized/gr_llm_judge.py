#!/usr/bin/env python3

import os
import sys
import json
import time
import anthropic
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

class GRLLMJudge:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.data_dir_personalized = os.path.join(self.data_dir, 'personalized')
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output')
        self.output_dir_personalized = os.path.join(self.output_dir, 'personalized')
        self.top_n_path = os.path.join(self.output_dir_personalized, 'gr_top_n_personalized.json')
        self.output_path = os.path.join(self.output_dir_personalized, 'gr_llm_judge_results.json')
        self.top_n_history = 10       # top N highest rated books for user history
        self.bottom_n_history = 10    # bottom N lowest rated books for user history
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def load(self):
        print('GRLLMJudge: loading data...')
        # load top n recommendations
        print('GRLLMJudge: loading top n recommendations...')
        self.recommendations = []
        with open(self.top_n_path, 'r') as f:
            for line in f:
                self.recommendations.append(json.loads(line))
        print(f'GRLLMJudge: loaded {len(self.recommendations)} recommendations.')

        # load train reviews for user history — only needed columns
        print('GRLLMJudge: loading train reviews for user history...')
        train_df = pd.read_parquet(
            os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet'),
            columns=['user_id_csv', 'book_id_csv', 'rating']
        )
        print(f'GRLLMJudge: loaded train reviews successfully. shape={train_df.shape}')

        # load book metadata for user history enrichment
        print('GRLLMJudge: loading book metadata...')
        books_meta = pd.read_parquet(
            os.path.join(self.data_dir, 'gr_books.parquet'),
            columns=['book_id', 'title', 'authors']
        ).drop_duplicates('book_id')
        print(f'GRLLMJudge: loaded book metadata successfully. num_books={len(books_meta)}')

        # load book id map for book_id_csv -> book_id lookup
        print('GRLLMJudge: loading book id map...')
        book_id_map = pd.read_csv(os.path.join(self.data_dir, 'gr_book_id_map.csv'))
        book_id_csv_to_book_id = book_id_map.set_index('book_id_csv')['book_id']
        print(f'GRLLMJudge: loaded book id map successfully.')

        # enrich train_df with title and authors
        print('GRLLMJudge: enriching train reviews with book metadata...')
        train_df['book_id'] = train_df['book_id_csv'].map(book_id_csv_to_book_id)
        train_df = train_df.merge(books_meta, on='book_id', how='left')
        train_df['title'] = train_df['title'].fillna('Unknown')
        train_df['authors'] = train_df['authors'].fillna('Unknown')

        # build user history lookup: user_id_csv -> df of (title, authors, rating)
        print('GRLLMJudge: building user history lookup...')
        self.user_history = {}
        for user_id_csv, group in train_df.groupby('user_id_csv'):
            self.user_history[user_id_csv] = group[[ 'book_id_csv', 'title', 'authors', 'rating']].copy()
        print(f'GRLLMJudge: built user history lookup for {len(self.user_history)} users.')

        print('GRLLMJudge: loaded data successfully.')

    def _get_user_history(self, user_id_csv):
        user_df = self.user_history.get(user_id_csv, pd.DataFrame())
        if user_df.empty:
            return [], []

        # top N highest rated — random tie-breaking, but the tie breaking is the same across runs as we fix random_state
        top_books = (
            user_df.sample(frac=1, random_state=42)
            .nlargest(self.top_n_history, 'rating')[['book_id_csv', 'title', 'authors', 'rating']]
            .to_dict(orient='records')
        )

        # get minimum rating in top books
        top_min_rating = min(b['rating'] for b in top_books) if top_books else 6

        # bottom N lowest rated — random tie-breaking, but the tie breaking is the same across runs as we fix random_state
        bottom_books = (
            user_df[user_df['rating'] < top_min_rating]
            .sample(frac=1, random_state=42)
            .nsmallest(self.bottom_n_history, 'rating')[['book_id_csv', 'title', 'authors', 'rating']]
            .to_dict(orient='records')
        )
        return top_books, bottom_books

    def _build_prompt(self, user_id_csv, cf_recommendations, hybrid_recommendations):
        top_books, bottom_books = self._get_user_history(user_id_csv)

        # format user history
        top_str = '\n'.join([
            f'{i+1}. {b["title"]} — {b["authors"]} (rated {int(b["rating"])})'
            for i, b in enumerate(top_books)
        ])
        bottom_str = '\n'.join([
            f'{i+1}. {b["title"]} — {b["authors"]} (rated {int(b["rating"])})'
            for i, b in enumerate(bottom_books)
        ]) if bottom_books else 'None available.'

        # format recommendations
        cf_str = '\n'.join([
            f'{i+1}. {r["title"]} — {r["authors"]}'
            for i, r in enumerate(cf_recommendations)
        ])
        hybrid_str = '\n'.join([
            f'{i+1}. {r["title"]} — {r["authors"]}'
            for i, r in enumerate(hybrid_recommendations)
        ])

        prompt = f"""You are evaluating book recommendations for a user based on their reading history.

Books this user LOVED (highest rated):
{top_str}

Books this user did NOT enjoy (lowest rated):
{bottom_str}

Now evaluate the following two recommendation lists for relevance to this user's taste.
Relevance measures how well the recommendations match what this user is likely to enjoy based on their history.

Recommendation List A (CF):
{cf_str}

Recommendation List B (Hybrid):
{hybrid_str}

Score each list on relevance to this user's taste on a scale of 1-10, where 1 is least relevant and 10 is most relevant
Respond ONLY with a JSON object, no preamble or explanation:
{{"cf_relevance": <score>, "hybrid_relevance": <score>}}"""

        return prompt

    def _call_claude(self, prompt):
        response = self.client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=100,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = response.content[0].text.strip()
        return json.loads(text)
    
    def _call_claude_with_retry(self, prompt, max_retries=5):
        for attempt in range(max_retries):
            try:
                return self._call_claude(prompt)
            except anthropic.RateLimitError as e:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f'GRLLMJudge: rate limit hit, retrying in {wait}s (attempt {attempt+1}/{max_retries})...')
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code == 529:  # overloaded
                    wait = 2 ** attempt
                    print(f'GRLLMJudge: API overloaded, retrying in {wait}s (attempt {attempt+1}/{max_retries})...')
                    time.sleep(wait)
                else:
                    raise e
        raise Exception(f'GRLLMJudge: max retries exceeded after {max_retries} attempts')

    def evaluate(self):
        print('GRLLMJudge: evaluating recommendations...')
        results = []
        start_time = time.time()
        # todo: implement incremental saving rather than holding all results and
        # flushing at the end, that way we don't incur claude API cost in next run if
        # the program crashed in the middle after few API requests!
        # since we are dealing with just 1000 results here, and our first execution succeded
        # and results are persisted, punting this for now!
        for i, rec in enumerate(self.recommendations):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                per_user = elapsed / i
                remaining = per_user * (len(self.recommendations) - i)
                print(f'GRLLMJudge: evaluating user {i+1}/{len(self.recommendations)}... elapsed={elapsed:.0f}s, per_user={per_user:.1f}s, eta={remaining:.0f}s')
            elif i == 0:
                print(f'GRLLMJudge: evaluating user {i+1}/{len(self.recommendations)}...')

            user_id_csv = rec['user_id_csv']
            cf_recommendations = rec['cf_recommendations']
            hybrid_recommendations = rec['hybrid_recommendations']

            try:
                prompt = self._build_prompt(user_id_csv, cf_recommendations, hybrid_recommendations)
                scores = self._call_claude_with_retry(prompt)
                results.append({
                    'user_id_csv': user_id_csv,
                    'cf_relevance': scores['cf_relevance'],
                    'hybrid_relevance': scores['hybrid_relevance']
                })
            except Exception as e:
                print(f'GRLLMJudge: error evaluating user {user_id_csv}: {e}')
                results.append({
                    'user_id_csv': user_id_csv,
                    'cf_relevance': None,
                    'hybrid_relevance': None
                })

        self.results = results
        # compute aggregate scores
        valid = [r for r in results if r['cf_relevance'] is not None]
        cf_mean = np.mean([r['cf_relevance'] for r in valid])
        hybrid_mean = np.mean([r['hybrid_relevance'] for r in valid])
        print(f'GRLLMJudge: evaluation complete. valid={len(valid)}/{len(results)}')
        print(f'GRLLMJudge: cf_mean_relevance={cf_mean:.2f}, hybrid_mean_relevance={hybrid_mean:.2f}')

    def save(self):
        print('GRLLMJudge: saving results...')
        valid = [r for r in self.results if r['cf_relevance'] is not None]
        cf_mean = round(float(np.mean([r['cf_relevance'] for r in valid])), 4)
        hybrid_mean = round(float(np.mean([r['hybrid_relevance'] for r in valid])), 4)

        output = {
            'cf_mean_relevance': cf_mean,
            'hybrid_mean_relevance': hybrid_mean,
            'num_valid': len(valid),
            'num_total': len(self.results),
            'results': self.results
        }
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'GRLLMJudge: saved to {self.output_path}')

    def display(self):
        print('\n========== GRLLMJudge Results ==========')
        with open(self.output_path, 'r') as f:
            output = json.load(f)
        print(f'CF Mean Relevance:     {output["cf_mean_relevance"]}')
        print(f'Hybrid Mean Relevance: {output["hybrid_mean_relevance"]}')
        print(f'Valid: {output["num_valid"]}/{output["num_total"]}')
        print('\n--- First 10 Results ---')
        for r in output['results'][:10]:
            print(f'  user={r["user_id_csv"]:>8}  cf={r["cf_relevance"]}  hybrid={r["hybrid_relevance"]}')
        print(f'================= Refer to {self.output_path} for all the results =======================\n')

    def run(self, override=False): # setting override=False as Hitting the LLM means spending $'s
        print(f'GRLLMJudge: starting with override={override}')
        if not override and os.path.exists(self.output_path):
            print('GRLLMJudge: output already exists, skipping. Use override=True to regenerate.')
            self.display()
            return
        self.load()
        self.evaluate()
        self.save()
        self.display()
        print('GRLLMJudge: done.')