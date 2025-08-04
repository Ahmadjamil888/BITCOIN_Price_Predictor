import React from 'react';
import { Newspaper, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { motion } from 'framer-motion';

interface NewsCardProps {
  sentiment: {
    sentiment_score: number;
    sentiment_label: string;
    timestamp: string;
  };
  loading?: boolean;
}

const NewsCard: React.FC<NewsCardProps> = ({ sentiment, loading = false }) => {
  const getSentimentColor = (label: string) => {
    switch (label.toLowerCase()) {
      case 'positive': return 'text-green-400 bg-green-900/20 border-green-400/30';
      case 'negative': return 'text-red-400 bg-red-900/20 border-red-400/30';
      case 'neutral': return 'text-yellow-400 bg-yellow-900/20 border-yellow-400/30';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-400/30';
    }
  };

  const getSentimentIcon = (label: string) => {
    switch (label.toLowerCase()) {
      case 'positive': return <TrendingUp className="w-5 h-5" />;
      case 'negative': return <TrendingDown className="w-5 h-5" />;
      case 'neutral': return <Minus className="w-5 h-5" />;
      default: return <Newspaper className="w-5 h-5" />;
    }
  };

  const getSentimentDescription = (score: number) => {
    if (score > 0.3) return "Very positive market sentiment detected";
    if (score > 0.1) return "Positive market sentiment detected";
    if (score > -0.1) return "Neutral market sentiment detected";
    if (score > -0.3) return "Negative market sentiment detected";
    return "Very negative market sentiment detected";
  };

  if (loading) {
    return (
      <div className="card">
        <h3 className="text-xl font-bold mb-6 gradient-text">News Sentiment Analysis</h3>
        <div className="flex justify-center items-center h-32">
          <div className="loading-dots">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div 
      className="card"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center space-x-3 mb-6">
        <Newspaper className="w-6 h-6 text-bitcoin-400" />
        <h3 className="text-xl font-bold gradient-text">News Sentiment Analysis</h3>
      </div>

      <div className="space-y-6">
        {/* Sentiment Score Visualization */}
        <div className="text-center">
          <div className="relative w-32 h-32 mx-auto mb-4">
            <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
              <circle
                cx="60"
                cy="60"
                r="50"
                stroke="#374151"
                strokeWidth="8"
                fill="none"
              />
              <circle
                cx="60"
                cy="60"
                r="50"
                stroke={sentiment.sentiment_label.toLowerCase() === 'positive' ? '#10b981' : 
                       sentiment.sentiment_label.toLowerCase() === 'negative' ? '#ef4444' : '#f59e0b'}
                strokeWidth="8"
                fill="none"
                strokeLinecap="round"
                strokeDasharray={`${Math.abs(sentiment.sentiment_score) * 314} 314`}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-2xl font-bold text-white">
                  {(sentiment.sentiment_score * 100).toFixed(0)}
                </div>
                <div className="text-xs text-gray-400">Score</div>
              </div>
            </div>
          </div>
        </div>

        {/* Sentiment Label */}
        <div className="flex justify-center">
          <div className={`flex items-center space-x-2 px-4 py-2 rounded-full border ${getSentimentColor(sentiment.sentiment_label)}`}>
            {getSentimentIcon(sentiment.sentiment_label)}
            <span className="font-semibold">{sentiment.sentiment_label}</span>
          </div>
        </div>

        {/* Description */}
        <div className="text-center">
          <p className="text-gray-300 text-sm">
            {getSentimentDescription(sentiment.sentiment_score)}
          </p>
        </div>

        {/* Sentiment Breakdown */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-green-900/20 rounded-lg border border-green-400/20">
            <div className="text-green-400 font-bold text-lg">
              {sentiment.sentiment_score > 0 ? Math.max(0, sentiment.sentiment_score * 100).toFixed(0) : '0'}%
            </div>
            <div className="text-xs text-gray-400">Positive</div>
          </div>
          <div className="text-center p-3 bg-yellow-900/20 rounded-lg border border-yellow-400/20">
            <div className="text-yellow-400 font-bold text-lg">
              {Math.abs(sentiment.sentiment_score) < 0.1 ? '100' : '0'}%
            </div>
            <div className="text-xs text-gray-400">Neutral</div>
          </div>
          <div className="text-center p-3 bg-red-900/20 rounded-lg border border-red-400/20">
            <div className="text-red-400 font-bold text-lg">
              {sentiment.sentiment_score < 0 ? Math.max(0, Math.abs(sentiment.sentiment_score) * 100).toFixed(0) : '0'}%
            </div>
            <div className="text-xs text-gray-400">Negative</div>
          </div>
        </div>

        {/* Impact on Price */}
        <div className="p-4 bg-dark-700 rounded-lg border border-dark-600">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Market Impact</span>
            <span className={`text-sm font-semibold ${
              sentiment.sentiment_score > 0.1 ? 'text-green-400' : 
              sentiment.sentiment_score < -0.1 ? 'text-red-400' : 'text-yellow-400'
            }`}>
              {sentiment.sentiment_score > 0.1 ? 'Bullish' : 
               sentiment.sentiment_score < -0.1 ? 'Bearish' : 'Neutral'}
            </span>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Last updated: {new Date(sentiment.timestamp).toLocaleString()}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default NewsCard;