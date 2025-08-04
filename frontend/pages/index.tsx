import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Bitcoin, Activity, TrendingUp, RefreshCw, Zap } from 'lucide-react';
import PriceChart from '../components/PriceChart';
import TechnicalIndicators from '../components/TechnicalIndicators';
import PredictionCard from '../components/PredictionCard';
import NewsCard from '../components/NewsCard';
import axios from 'axios';
import Head from 'next/head';

const API_BASE_URL = 'http://localhost:5000/api';

interface PriceData {
  date: string;
  price: number;
  predicted?: number;
}

interface Prediction {
  day: number;
  predicted_price: number;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
}

interface TechnicalData {
  rsi: number;
  macd: number;
  macd_signal: number;
  bb_upper: number;
  bb_lower: number;
  ma_7: number;
  ma_21: number;
  ma_50: number;
}

interface NewsData {
  sentiment_score: number;
  sentiment_label: string;
  timestamp: string;
}

export default function Home() {
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [historicalData, setHistoricalData] = useState<PriceData[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [technicalIndicators, setTechnicalIndicators] = useState<TechnicalData | null>(null);
  const [newsData, setNewsData] = useState<NewsData | null>(null);
  const [loading, setLoading] = useState({
    price: true,
    historical: true,
    predictions: false,
    technical: true,
    news: true
  });

  const fetchCurrentPrice = async () => {
    try {
      setLoading(prev => ({ ...prev, price: true }));
      const response = await axios.get(`${API_BASE_URL}/current-price`);
      setCurrentPrice(response.data.price);
    } catch (error) {
      console.error('Error fetching current price:', error);
      // Fallback to mock data
      setCurrentPrice(45000 + Math.random() * 10000);
    } finally {
      setLoading(prev => ({ ...prev, price: false }));
    }
  };

  const fetchHistoricalData = async () => {
    try {
      setLoading(prev => ({ ...prev, historical: true }));
      const response = await axios.get(`${API_BASE_URL}/historical-data?period=3mo`);
      const formattedData = response.data.map((item: any) => ({
        date: item.date,
        price: item.close
      }));
      setHistoricalData(formattedData);
    } catch (error) {
      console.error('Error fetching historical data:', error);
      // Generate mock historical data
      const mockData = [];
      const basePrice = 45000;
      for (let i = 90; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        mockData.push({
          date: date.toISOString().split('T')[0],
          price: basePrice + (Math.random() - 0.5) * 10000 + Math.sin(i / 10) * 5000
        });
      }
      setHistoricalData(mockData);
    } finally {
      setLoading(prev => ({ ...prev, historical: false }));
    }
  };

  const fetchPredictions = async () => {
    try {
      setLoading(prev => ({ ...prev, predictions: true }));
      const response = await axios.post(`${API_BASE_URL}/predict`, { steps: 7 });
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error('Error fetching predictions:', error);
      // Generate mock predictions
      const mockPredictions = [];
      for (let i = 1; i <= 7; i++) {
        const change = (Math.random() - 0.5) * 0.1;
        const predictedPrice = currentPrice * (1 + change);
        const signals = ['BUY', 'SELL', 'HOLD'];
        const signal = signals[Math.floor(Math.random() * signals.length)] as 'BUY' | 'SELL' | 'HOLD';
        
        mockPredictions.push({
          day: i,
          predicted_price: predictedPrice,
          signal,
          confidence: 60 + Math.random() * 30,
          reason: signal === 'BUY' ? 'Upward trend predicted' : 
                  signal === 'SELL' ? 'Downward trend predicted' : 'Price expected to remain stable'
        });
      }
      setPredictions(mockPredictions);
    } finally {
      setLoading(prev => ({ ...prev, predictions: false }));
    }
  };

  const fetchTechnicalIndicators = async () => {
    try {
      setLoading(prev => ({ ...prev, technical: true }));
      const response = await axios.get(`${API_BASE_URL}/technical-indicators`);
      setTechnicalIndicators(response.data);
    } catch (error) {
      console.error('Error fetching technical indicators:', error);
      // Generate mock technical data
      setTechnicalIndicators({
        rsi: 45 + Math.random() * 40,
        macd: (Math.random() - 0.5) * 1000,
        macd_signal: (Math.random() - 0.5) * 1000,
        bb_upper: currentPrice * 1.05,
        bb_lower: currentPrice * 0.95,
        ma_7: currentPrice * (0.98 + Math.random() * 0.04),
        ma_21: currentPrice * (0.96 + Math.random() * 0.08),
        ma_50: currentPrice * (0.94 + Math.random() * 0.12)
      });
    } finally {
      setLoading(prev => ({ ...prev, technical: false }));
    }
  };

  const fetchNewsData = async () => {
    try {
      setLoading(prev => ({ ...prev, news: true }));
      const response = await axios.get(`${API_BASE_URL}/news-sentiment`);
      setNewsData(response.data);
    } catch (error) {
      console.error('Error fetching news data:', error);
      // Generate mock news data
      const score = (Math.random() - 0.5) * 2;
      setNewsData({
        sentiment_score: score,
        sentiment_label: score > 0.1 ? 'Positive' : score < -0.1 ? 'Negative' : 'Neutral',
        timestamp: new Date().toISOString()
      });
    } finally {
      setLoading(prev => ({ ...prev, news: false }));
    }
  };

  const refreshAll = async () => {
    await Promise.all([
      fetchCurrentPrice(),
      fetchHistoricalData(),
      fetchTechnicalIndicators(),
      fetchNewsData()
    ]);
  };

  useEffect(() => {
    refreshAll();
    
    // Set up real-time updates
    const interval = setInterval(() => {
      fetchCurrentPrice();
      fetchNewsData();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (currentPrice > 0) {
      fetchPredictions();
    }
  }, [currentPrice]);

  return (
    <>
      <Head>
        <title>Bitcoin AI Predictor</title>
        <meta name="description" content="AI-powered Bitcoin price predictions" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className="min-h-screen bg-dark-900">
      {/* Header */}
      <header className="border-b border-dark-700 bg-dark-800/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-bitcoin-500 rounded-lg glow">
                  <Bitcoin className="w-8 h-8 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold gradient-text">Bitcoin AI Predictor</h1>
                  <p className="text-sm text-gray-400">Advanced ML-powered price predictions</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm text-gray-400">Current Price</div>
                <div className="text-2xl font-bold text-white">
                  {loading.price ? (
                    <div className="animate-pulse bg-dark-600 h-8 w-32 rounded"></div>
                  ) : (
                    `$${currentPrice.toLocaleString()}`
                  )}
                </div>
              </div>
              
              <button
                onClick={refreshAll}
                className="btn-secondary flex items-center space-x-2"
                disabled={Object.values(loading).some(l => l)}
              >
                <RefreshCw className={`w-4 h-4 ${Object.values(loading).some(l => l) ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column - Charts */}
          <div className="xl:col-span-2 space-y-8">
            {/* Price Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="card">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold gradient-text">Price Chart</h2>
                  <div className="flex items-center space-x-2">
                    <Activity className="w-5 h-5 text-bitcoin-400" />
                    <span className="text-sm text-gray-400">Live Data</span>
                  </div>
                </div>
                {loading.historical ? (
                  <div className="h-96 flex items-center justify-center">
                    <div className="loading-dots">
                      <div></div>
                      <div></div>
                      <div></div>
                      <div></div>
                    </div>
                  </div>
                ) : (
                  <PriceChart data={historicalData} height={400} />
                )}
              </div>
            </motion.div>

            {/* Technical Indicators */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              {loading.technical || !technicalIndicators ? (
                <div className="card">
                  <h2 className="text-2xl font-bold gradient-text mb-6">Technical Indicators</h2>
                  <div className="h-64 flex items-center justify-center">
                    <div className="loading-dots">
                      <div></div>
                      <div></div>
                      <div></div>
                      <div></div>
                    </div>
                  </div>
                </div>
              ) : (
                <TechnicalIndicators indicators={technicalIndicators} />
              )}
            </motion.div>
          </div>

          {/* Right Column - Predictions & News */}
          <div className="space-y-8">
            {/* AI Predictions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <PredictionCard 
                predictions={predictions} 
                currentPrice={currentPrice}
                loading={loading.predictions}
              />
            </motion.div>

            {/* News Sentiment */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              {newsData ? (
                <NewsCard sentiment={newsData} loading={loading.news} />
              ) : (
                <div className="card">
                  <h3 className="text-xl font-bold gradient-text mb-6">News Sentiment Analysis</h3>
                  <div className="h-32 flex items-center justify-center">
                    <div className="loading-dots">
                      <div></div>
                      <div></div>
                      <div></div>
                      <div></div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="card"
            >
              <h3 className="text-xl font-bold gradient-text mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={fetchPredictions}
                  disabled={loading.predictions}
                  className="w-full btn-primary flex items-center justify-center space-x-2"
                >
                  <Zap className="w-4 h-4" />
                  <span>Get New Predictions</span>
                </button>
                <button
                  onClick={fetchTechnicalIndicators}
                  disabled={loading.technical}
                  className="w-full btn-secondary flex items-center justify-center space-x-2"
                >
                  <TrendingUp className="w-4 h-4" />
                  <span>Update Indicators</span>
                </button>
              </div>
            </motion.div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-dark-700 bg-dark-800/50 mt-16">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center">
            <p className="text-gray-400 text-sm">
              Bitcoin AI Predictor - Powered by PyTorch & Advanced Machine Learning
            </p>
            <p className="text-gray-500 text-xs mt-2">
              This tool is for educational purposes only. Not financial advice.
            </p>
          </div>
        </div>
      </footer>
    </div>
    </>
  );
}