import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadialBarChart, RadialBar, PieChart, Pie, Cell } from 'recharts';

interface TechnicalIndicatorsProps {
  indicators: {
    rsi: number;
    macd: number;
    macd_signal: number;
    bb_upper: number;
    bb_lower: number;
    ma_7: number;
    ma_21: number;
    ma_50: number;
  };
}

const TechnicalIndicators: React.FC<TechnicalIndicatorsProps> = ({ indicators }) => {
  const rsiData = [
    { name: 'RSI', value: indicators.rsi, fill: indicators.rsi > 70 ? '#ef4444' : indicators.rsi < 30 ? '#10b981' : '#f59e0b' }
  ];

  const maData = [
    { name: 'MA 7', value: indicators.ma_7, fill: '#8b5cf6' },
    { name: 'MA 21', value: indicators.ma_21, fill: '#06b6d4' },
    { name: 'MA 50', value: indicators.ma_50, fill: '#f59e0b' }
  ];

  const getRSISignal = (rsi: number) => {
    if (rsi > 70) return { signal: 'Overbought', color: 'text-red-400', bg: 'bg-red-900/20' };
    if (rsi < 30) return { signal: 'Oversold', color: 'text-green-400', bg: 'bg-green-900/20' };
    return { signal: 'Neutral', color: 'text-yellow-400', bg: 'bg-yellow-900/20' };
  };

  const getMACDSignal = (macd: number, signal: number) => {
    if (macd > signal) return { signal: 'Bullish', color: 'text-green-400', bg: 'bg-green-900/20' };
    return { signal: 'Bearish', color: 'text-red-400', bg: 'bg-red-900/20' };
  };

  const rsiSignal = getRSISignal(indicators.rsi);
  const macdSignal = getMACDSignal(indicators.macd, indicators.macd_signal);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* RSI Indicator */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4 gradient-text">RSI (Relative Strength Index)</h3>
        <div className="flex items-center justify-between mb-4">
          <div className={`px-3 py-1 rounded-full text-sm font-semibold ${rsiSignal.bg} ${rsiSignal.color}`}>
            {rsiSignal.signal}
          </div>
          <div className="text-2xl font-bold text-white">
            {indicators.rsi.toFixed(1)}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={rsiData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9ca3af" />
            <YAxis domain={[0, 100]} stroke="#9ca3af" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1f2937', 
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* MACD Indicator */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4 gradient-text">MACD</h3>
        <div className="flex items-center justify-between mb-4">
          <div className={`px-3 py-1 rounded-full text-sm font-semibold ${macdSignal.bg} ${macdSignal.color}`}>
            {macdSignal.signal}
          </div>
          <div className="text-right">
            <div className="text-lg font-bold text-white">
              {indicators.macd.toFixed(2)}
            </div>
            <div className="text-sm text-gray-400">
              Signal: {indicators.macd_signal.toFixed(2)}
            </div>
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">MACD Line</span>
            <span className="text-sm font-semibold text-blue-400">{indicators.macd.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Signal Line</span>
            <span className="text-sm font-semibold text-orange-400">{indicators.macd_signal.toFixed(2)}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Histogram</span>
            <span className={`text-sm font-semibold ${(indicators.macd - indicators.macd_signal) > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {(indicators.macd - indicators.macd_signal).toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      {/* Moving Averages */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4 gradient-text">Moving Averages</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={maData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" tickFormatter={(value) => `$${value.toLocaleString()}`} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1f2937', 
                border: '1px solid #374151',
                borderRadius: '8px'
              }}
              formatter={(value: any) => [`$${value.toLocaleString()}`, 'Price']}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Bollinger Bands */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4 gradient-text">Bollinger Bands</h3>
        <div className="space-y-4">
          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
            <span className="text-sm text-gray-400">Upper Band</span>
            <span className="text-lg font-bold text-red-400">${indicators.bb_upper.toLocaleString()}</span>
          </div>
          <div className="flex justify-between items-center p-3 bg-dark-700 rounded-lg">
            <span className="text-sm text-gray-400">Lower Band</span>
            <span className="text-lg font-bold text-green-400">${indicators.bb_lower.toLocaleString()}</span>
          </div>
          <div className="text-center p-3 bg-dark-700 rounded-lg">
            <div className="text-sm text-gray-400 mb-1">Band Width</div>
            <div className="text-lg font-bold text-bitcoin-400">
              ${(indicators.bb_upper - indicators.bb_lower).toLocaleString()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TechnicalIndicators;