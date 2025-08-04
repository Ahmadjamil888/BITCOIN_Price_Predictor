import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

interface PriceData {
  date: string;
  price: number;
  predicted?: number;
}

interface PriceChartProps {
  data: PriceData[];
  showPrediction?: boolean;
  height?: number;
}

const PriceChart: React.FC<PriceChartProps> = ({ data, showPrediction = false, height = 400 }) => {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-dark-800 p-4 rounded-lg border border-dark-600 shadow-xl">
          <p className="text-gray-300 text-sm">{`Date: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {`${entry.dataKey === 'price' ? 'Price' : 'Predicted'}: $${entry.value?.toLocaleString()}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f97316" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#f97316" stopOpacity={0.1}/>
            </linearGradient>
            <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="date" 
            stroke="#9ca3af"
            fontSize={12}
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis 
            stroke="#9ca3af"
            fontSize={12}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="price"
            stroke="#f97316"
            strokeWidth={2}
            fill="url(#priceGradient)"
          />
          {showPrediction && (
            <Area
              type="monotone"
              dataKey="predicted"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="5 5"
              fill="url(#predictedGradient)"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PriceChart;