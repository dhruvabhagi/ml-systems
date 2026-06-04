#!/usr/bin/env python3
import os
from gr_reviews_sentiment_analyzer import GoodreadsReviewsSentimentAnalyzer
from gr_top_n_global import GRTopNGlobal
from personalized.gr_personalized_recommend_data_prep import GRPersonalizedRecommendDataPrep
from personalized.gr_collaborative_filtering_model import GRCollaborativeFilteringModel
from personalized.gr_book_content_feature_builder import GRBookContentFeatureBuilder
from personalized.gr_hybrid_model import GRHybridModel
from personalized.gr_top_n_personalized import GRTopNPersonalized
from personalized.gr_llm_judge import GRLLMJudge
from personalized.gr_segmentation_analysis import GRSegmentationAnalysis

def _dir_setup():
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'personalized'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'personalized'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'personalized'), exist_ok=True)

if __name__ == "__main__":

    _dir_setup()    

    # step 1: Score the sentitments for of 15.7M reviews on 2.08M books by 465K users!
    analyzer = GoodreadsReviewsSentimentAnalyzer()
    analyzer.run()

    # step 2: predict top n gloabl books from 2.08M books based on overall user sentiment
    top_n_global = GRTopNGlobal()
    top_n_global.run()

    # step 3: Data preparation for personalized recommendation model
    data_prep = GRPersonalizedRecommendDataPrep()
    data_prep.run()

    # step 4: collaborative filtering    
    cf_model = GRCollaborativeFilteringModel()
    cf_model.run()

    # step 5: book and author content feature building for consumption by downstream hybrid neural network
    book_content_feature_builder = GRBookContentFeatureBuilder()
    book_content_feature_builder.run()

    # step 6: hybrid neural network — combines pretrained collaborative filtering embeddings
    # with author embeddings and content features for improved recommendation quality
    hybrid_model = GRHybridModel()
    hybrid_model.run()

    # step 7: generate top N personalized book recommendations for K users with most varied book ratings
    top_n_personalized = GRTopNPersonalized()
    top_n_personalized.run()

    # step 8: LLM as a judge on the personalized recommendataions
    llm_judge = GRLLMJudge()
    llm_judge.run()

    # step 9: perform segmentation analysis on the top N users with personalized recommendataions
    segmentation = GRSegmentationAnalysis()
    segmentation.run()