//+------------------------------------------------------------------+
//|                     GMM ONNX Trading EA                          |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>
#include <Math\Alglib\alglib.mqh>

CTrade trade;

#resource "gmm.onnx" as uchar GMMModel[]

// === INPUTS ===
input double InpLot = 0.1;
input double InpThreshold = 0.60;
input int InpMagic = 123456;
input bool InpLog = false;

// mapping cluster -> action
// IMPORTANT: trebuie sa fie identic cu mapping.json
input int Cluster0 = 1;   // BUY
input int Cluster1 = -1;  // SELL
input int Cluster2 = 0;   // FLAT

// === ONNX ===
long onnx_handle = INVALID_HANDLE;

// === FEATURE BUFFER ===
double features[10];
#include <Math\Alglib\alglib.mqh>

// sau direct vectorf / matrixf dacă nu ai nevoie de alglib

// GLOBAL
vectorf input_data;
vectorf output_data;
const long input_shape[] = {1,10};
const long output_shape[] = {1,3};


// INIT
int OnInit()
{
   if (InpLog)
   {
      Print("=== INIT START ===");   
   }
   
   onnx_handle = OnnxCreateFromBuffer(GMMModel, ONNX_DEFAULT);

   if(onnx_handle == INVALID_HANDLE)
   {
      if (InpLog)
      {
         Print("ONNX load failed from buffer");
         Print("Error: ", GetLastError());
      }
      return INIT_FAILED;
   }


   if (InpLog)
      Print("ONNX LOAD OK");
   
   OnnxSetInputShape(onnx_handle, 0, input_shape);
   
   if (InpLog)
      Print("ONNX OnnxSetInputShape OK");
   
   OnnxSetOutputShape(onnx_handle, 0, output_shape);
   
   if (InpLog)
      Print("ONNX OnnxSetOutputShape OK");
   
   input_data.Resize(10);     // 10 features
   if (InpLog)
      Print("Input resize OK");
      
   output_data.Resize(3);     // 3 clusters
   if (InpLog)
      Print("Output resize OK");

   return INIT_SUCCEEDED;
}


// === DEINIT ===
void OnDeinit(const int reason)
{
   if(onnx_handle != INVALID_HANDLE)
      OnnxRelease(onnx_handle);
}

// === FEATURE ENGINEERING ===
void BuildFeatures(double &out[])
{
   int shift = 0;

   double close0 = iClose(_Symbol, PERIOD_M15, shift);
   double close1 = iClose(_Symbol, PERIOD_M15, shift+1);
   double close3 = iClose(_Symbol, PERIOD_M15, shift+3);
   double close5 = iClose(_Symbol, PERIOD_M15, shift+5);
   double close10 = iClose(_Symbol, PERIOD_M15, shift+10);

   out[0] = (close0 / close1) - 1.0;
   out[1] = (close0 / close3) - 1.0;
   out[2] = (close0 / close5) - 1.0;
   out[3] = (close0 / close10) - 1.0;

   // volatility
   double sum = 0;
   double mean = 0;
   for(int i=1;i<=10;i++)
      mean += (iClose(_Symbol, PERIOD_M15, i) - close0);
   mean /= 10;

   for(int i=1;i<=10;i++)
   {
      double r = (iClose(_Symbol, PERIOD_M15, i) - close0);
      sum += (r-mean)*(r-mean);
   }
   out[4] = MathSqrt(sum/10);

   out[5] = out[4]; // simplificat vol_20

   // SMA
   double sma10=0, sma20=0;
   for(int i=0;i<10;i++) sma10 += iClose(_Symbol, PERIOD_M15, i);
   for(int i=0;i<20;i++) sma20 += iClose(_Symbol, PERIOD_M15, i);

   sma10/=10;
   sma20/=20;

   out[6] = (close0 / sma10) - 1;
   out[7] = (close0 / sma20) - 1;

   // zscore
   double mean20=0, std20=0;
   for(int i=0;i<20;i++) mean20 += iClose(_Symbol, PERIOD_M15, i);
   mean20/=20;

   for(int i=0;i<20;i++)
      std20 += MathPow(iClose(_Symbol, PERIOD_M15, i)-mean20,2);

   std20 = MathSqrt(std20/20);
   out[8] = (close0 - mean20)/std20;

   // ATR approx
   double high = iHigh(_Symbol, PERIOD_M15, 0);
   double low = iLow(_Symbol, PERIOD_M15, 0);
   out[9] = (high - low)/close0;
}

// === CLUSTER → ACTION ===
int MapCluster(int cluster)
{
   if(cluster == 0) return Cluster0;
   if(cluster == 1) return Cluster1;
   if(cluster == 2) return Cluster2;
   return 0;
}

// === ONNX PREDICT ===
int Predict(double &confidence)
{
   for(int i=0;i<10;i++)
      input_data[i] = (float)features[i];

   if(!OnnxRun(onnx_handle, ONNX_NO_CONVERSION, input_data, output_data))
   {
      Print("ONNX run failed ", GetLastError());
      return 0;
   }

   int best = 0;
   double maxv = output_data[0];

   for(int i=1;i<3;i++)
   {
      if(output_data[i] > maxv)
      {
         maxv = output_data[i];
         best = i;
      }
   }

   confidence = maxv;
   return best;
}

// === TRADING ===
void ExecuteTrade(int signal)
{
   if(PositionSelect(_Symbol))
      return;

   if(signal == 1)
      trade.Buy(InpLot, _Symbol);

   if(signal == -1)
      trade.Sell(InpLot, _Symbol);
}

// === MAIN LOOP ===
void OnTick()
{
   int bars = Bars(_Symbol, PERIOD_M15);
   
   if(bars < 50)
   {
      if(InpLog)
         Print("Not enough bars: ", bars);
      return;
   }
   
   static datetime last_bar = 0;

   datetime current_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(current_bar == last_bar)
      return;

   last_bar = current_bar;

   BuildFeatures(features);

   double conf;
   int cluster = Predict(conf);

   if(conf < InpThreshold)
      return;

   int signal = MapCluster(cluster);

   if(signal != 0)
      ExecuteTrade(signal);

   if (InpLog)
      Print("Cluster=", cluster, " Confidence=", conf, " Signal=", signal);
}

double OnTester()
{
   double profit        = TesterStatistics(STAT_PROFIT);
   double pf            = TesterStatistics(STAT_PROFIT_FACTOR);
   double recovery      = TesterStatistics(STAT_RECOVERY_FACTOR);
   double dd_percent    = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   double trades        = TesterStatistics(STAT_TRADES);

   // Penalty if there are too few transactions
   double trade_penalty = 1.0;
   if(trades < 20)
      trade_penalty = 0.25;
   else if(trades < 50)
      trade_penalty = 0.60;

   // Robust score, not only brut profit
   double score = 0.0;

   if(dd_percent >= 0.0)
      score = (profit * MathMax(pf, 0.01) * MathMax(recovery, 0.01) * trade_penalty) / (1.0 + dd_percent);

   return score;
}
