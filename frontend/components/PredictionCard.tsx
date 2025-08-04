import React from 'react';
import { TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';

interface Prediction {
  day: number;
  predicted_price: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
}

interface PredictionCardProps {
  predictions: Prediction[];
  currentPrice: number;
  loading?: boolean;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ predictions, currentPrice, loading = false }) => {
  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-400 bg-green-900/20 border-green-400/30';
      case 'SELL': return 'text-red-400 bg-red-900/20 border-red-400/30';
      case 'HOLD': return 'text-yellow-400 bg-yellow-900/20 border-yellow-400/30';
      default: return 'text-gray-400 bg-gray-900/20 border-gray-400/30';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <TrendingUp className="w-5 h-5" />;
      case 'SELL': return <TrendingDown className="w-5 h-5" />;
      case 'HOLD': return <Minus className="w-5 h-5" />;
      default: return <AlertTriangle className="w-5 h-5" />;
    }
  };

  const getPriceChange = (predictedPrice: number) => {
    const change = ((predictedPrice - currentPrice) / currentPrice) * 100;
    return {
      value: change,
      color: change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-gray-400',
      sign: change > 0 ? '+' : ''
    };
  };

  if (loading) {
    return (
      <div className="card">
        <h3 className="text-xl font-bold mb-6 gradient-text">AI Price Predictions</h3>
        <div className="flex justify-center items-center h-40">
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
    <div className="card">
      <h3 className="text-xl font-bold mb-6 gradient-text">AI Price Predictions</h3>
      
      <div className="mb-6 p-4 bg-dark-700 rounded-lg border border-dark-600">
        <div className="text-sm text-gray-400 mb-1">Current Bitcoin Price</div>
        <div className="text-2xl font-bold text-white">${currentPrice.toLocaleString()}</div>
      </div>

      <div className="space-y-4">
        {predictions.map((prediction, index) => {
          const priceChange = getPriceChange(prediction.predicted_price);
          
          return (
            <motion.div
              key={prediction.day}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="p-4 bg-dark-700 rounded-lg border border-dark-600 hover:border-bitcoin-500/50 transition-all duration-200"
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-3">
                  <div className={`flex items-center space-x-2 px-3 py-1 rounded-full border ${getSignalColor(prediction.signal)}`}>
                    {getSignalIcon(prediction.signal)}
                    <span className="font-semibold text-sm">{prediction.signal}</span>
                  </div>
                  <div className="text-sm text-gray-400">
                    Day {prediction.day}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-white">
                    ${prediction.predicted_price.toLocaleString()}
                  </div>
                  <div className={`text-sm font-semibold ${priceChange.color}`}>
                    {priceChange.sign}{priceChange.value.toFixed(2)}%
                  </div>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-400">
                  {prediction.reason}
                </div>
                <div className="flex items-center space-x-2">
                  <div className="text-xs text-gray-500">Confidence:</div>
                  <div className="flex items-center space-x-1">
                    <div className="w-16 h-2 bg-dark-600 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-bitcoin-500 to-bitcoin-400 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(prediction.confidence, 100)}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400">{prediction.confidence.toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="mt-6 p-4 bg-gradient-to-r from-bitcoin-900/20 to-bitcoin-800/20 rounded-lg border border-bitcoin-500/20">
        <div className="flex items-center space-x-2 mb-2">
          <AlertTriangle className="w-4 h-4 text-bitcoin-400" />
          <span className="text-sm font-semibold text-bitcoin-400">Disclaimer</span>
        </div>
        <p className="text-xs text-gray-400">
          These predictions are generated by AI and should not be considered as financial advice. 
          Cryptocurrency investments are highly volatile and risky. Always do your own research.
        </p>
      </div>
    </div>
  );
};

export default PredictionCard;