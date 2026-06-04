#!/usr/bin/env python3

import os
import json
import pandas as pd
from scipy import stats

class GRSegmentationAnalysis:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.data_dir_personalized = os.path.join(self.data_dir, 'personalized')
        self.model_dir_personalized = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model', 'personalized')
        self.output_dir_personalized = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output', 'personalized')
        self.top_n_path = os.path.join(self.output_dir_personalized, 'gr_top_n_personalized.json')
        self.llm_results_path = os.path.join(self.output_dir_personalized, 'gr_llm_judge_results.json')
        self.output_path = os.path.join(self.output_dir_personalized, 'gr_segmentation_analysis.json')

    def load(self):
        print('GRSegmentationAnalysis: loading data...')

        # load llm judge results
        print('GRSegmentationAnalysis: loading llm judge results...')
        with open(self.llm_results_path, 'r') as f:
            llm_output = json.load(f)
        self.llm_results = {r['user_id_csv']: r for r in llm_output['results'] if r['cf_relevance'] is not None}
        print(f'GRSegmentationAnalysis: loaded {len(self.llm_results)} valid llm results.')

        # load top_n to get the 1000 user ids
        print('GRSegmentationAnalysis: loading top n recommendations...')
        self.user_ids = []
        with open(self.top_n_path, 'r') as f:
            for line in f:
                rec = json.loads(line)
                self.user_ids.append(rec['user_id_csv'])
        print(f'GRSegmentationAnalysis: loaded {len(self.user_ids)} users.')

        # load train reviews for the 1000 users only
        print('GRSegmentationAnalysis: loading train reviews...')
        train_df = pd.read_parquet(
            os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet'),
            columns=['user_id_csv', 'book_id_csv']
        )
        train_df = train_df[train_df['user_id_csv'].isin(self.user_ids)]
        print(f'GRSegmentationAnalysis: loaded train reviews. shape={train_df.shape}')

        # load review_count per book from content features
        print('GRSegmentationAnalysis: loading book content features...')
        content_df = pd.read_parquet(
            os.path.join(self.model_dir_personalized, 'gr_book_content_features.parquet'),
            columns=['book_id_csv', 'review_count']
        )
        print(f'GRSegmentationAnalysis: loaded content features. shape={content_df.shape}')

        # compute mainstream score per user = average review_count of books they rated
        # so the user_mainstream_scores is the average of review_count's of the BOOKS
        # read by the user! These review counts capture all the reviews by all the users for that book!
        # This is not the reviews at a particular (current) user level
        # We are interestd in finding users who reads books whose review counts are less
        # to see if hybrid model plays a better role in recommendataion!
        print('GRSegmentationAnalysis: computing mainstream scores...')
        train_df = train_df.merge(content_df, on='book_id_csv', how='left')
        self.user_mainstream_scores = (
            train_df.groupby('user_id_csv')['review_count']
            .mean()
            .to_dict()
        )
        print(f'GRSegmentationAnalysis: computed mainstream scores for {len(self.user_mainstream_scores)} users.')
        print('GRSegmentationAnalysis: loaded data successfully.')

    def analyze(self):
        print('GRSegmentationAnalysis: analyzing...')

        # build per-user dataframe
        rows = []
        for user_id_csv in self.user_ids:
            if user_id_csv not in self.llm_results:
                continue
            if user_id_csv not in self.user_mainstream_scores:
                continue
            rows.append({
                'user_id_csv': user_id_csv,
                'mainstream_score': self.user_mainstream_scores[user_id_csv],
                'cf_relevance': self.llm_results[user_id_csv]['cf_relevance'],
                'hybrid_relevance': self.llm_results[user_id_csv]['hybrid_relevance'],
                'relevance_delta': self.llm_results[user_id_csv]['hybrid_relevance'] - self.llm_results[user_id_csv]['cf_relevance']
            })

        df = pd.DataFrame(rows)
        print(f'GRSegmentationAnalysis: built dataframe. shape={df.shape}')

        # define thresholds using 25th and 75th percentile
        p25 = df['mainstream_score'].quantile(0.25)
        p75 = df['mainstream_score'].quantile(0.75)
        print(f'GRSegmentationAnalysis: mainstream_score p25={p25:.1f}, p75={p75:.1f}')

        # split into niche and mainstream
        niche_df = df[df['mainstream_score'] <= p25]
        mainstream_df = df[df['mainstream_score'] >= p75]
        middle_df = df[(df['mainstream_score'] > p25) & (df['mainstream_score'] < p75)]

        print(f'GRSegmentationAnalysis: niche={len(niche_df)}, mainstream={len(mainstream_df)}, middle={len(middle_df)}')

        # compute stats per group
        def group_stats(group_df, name):
            cf_mean = group_df['cf_relevance'].mean()
            hybrid_mean = group_df['hybrid_relevance'].mean()
            relevance_delta_mean = group_df['relevance_delta'].mean()
            t_stat, p_value = stats.ttest_rel(group_df['hybrid_relevance'], group_df['cf_relevance'])
            # t=mean(relevance_delta_mean) / (std(relevance_delta_mean)/sqrt(n))
            # higher t means, mean is higher, but std deviation is lower
            # that means most of the values are closer to the mean, i.e majority of the nice users
            # have the relevance_delta more or less closer to the mean
            # so the probability that our finding (that hybrid beats CF for niche users) to be a random chance is less
            # i.e higher t -> lower p -> less proabable that our finding is random -> so our finding holds true for majority/all niche users
            return {
                'group': name,
                'n': len(group_df),
                'cf_mean': round(cf_mean, 4),
                'hybrid_mean': round(hybrid_mean, 4),
                'relevance_delta_mean': round(relevance_delta_mean, 4),
                't_stat': round(float(t_stat), 4),
                'p_value': round(float(p_value), 4)
            }

        self.niche_stats = group_stats(niche_df, 'niche')
        self.mainstream_stats = group_stats(mainstream_df, 'mainstream')
        self.overall_stats = group_stats(df, 'overall')

        print('GRSegmentationAnalysis: analysis complete.')

    def display(self):
        print('\n========== GRSegmentationAnalysis Results ==========')
        print(f'{"Group":<15} {"N":<8} {"CF Mean":<12} {"Hybrid Mean":<15} {"Delta Mean":<12} {"t-stat":<10} {"p-value":<10}')
        print('-' * 82)
        with open(self.output_path, 'r') as f:
            output = json.load(f)
        for stats_row in [output['niche'], output['mainstream'], output['overall']]:
            sig = '***' if stats_row['p_value'] < 0.001 else '**' if stats_row['p_value'] < 0.01 else '*' if stats_row['p_value'] < 0.05 else 'ns'
            print(f'{stats_row["group"]:<15} {stats_row["n"]:<8} {stats_row["cf_mean"]:<12} {stats_row["hybrid_mean"]:<15} {stats_row["relevance_delta_mean"]:<12} {stats_row["t_stat"]:<10} {stats_row["p_value"]:<10} {sig}')
        print('* p<0.05  ** p<0.01  *** p<0.001  ns=not significant')
        print('=====================================================\n')

    def save(self):
        print('GRSegmentationAnalysis: saving results...')
        output = {
            'niche': self.niche_stats,
            'mainstream': self.mainstream_stats,
            'overall': self.overall_stats
        }
        with open(self.output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'GRSegmentationAnalysis: saved to {self.output_path}')

    def run(self, override=False):
        print(f'GRSegmentationAnalysis: starting with override={override}')
        if not override and os.path.exists(self.output_path):
            print('GRSegmentationAnalysis: output already exists, skipping. Use override=True to regenerate.')
            self.display()
            return
        self.load()
        self.analyze()
        self.display()
        self.save()
        print('GRSegmentationAnalysis: done.')