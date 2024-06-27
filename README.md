<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> SugarNet: Personalized Blood Glucose Forecast for
Diabetic Patients with Joint Time-Frequency Learning </b></h2>
</div>

🌟**SugarNet** is a novel deep learning model for
personalized BG forecasts based on food, medication, insulin, and BG history. The
model employs a join time-frequency domain learner with a convlutional LSTM.


🌟**High level architecture** 

<p align="center">
<img src="./pic/overall.png"  alt="" align=center />
</p>


🌟**Baselines**

The following models are compared with SugarNet.

1. [**TimeMixer**]: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting[\[ICLR 2024\]](https://openreview.net/pdf?id=7oLshfEIC2)[\[GitHub Repo\]](https://github.com/KimMeen/Time-LLM)

2. [**iTransformer**]: Inverted Transformers Are Effective for Time Series Forecasting[\[ICLR 2024\]](https://arxiv.org/abs/2310.06625)
[\[GitHub Repo\]](https://github.com/thuml/iTransformer)

3. [**PatchTST**]: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers[\[ICLR 2023\]](https://openreview.net/pdf?id=Jbdc0vTOcol)[\[GitHub Repo\]](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)

4. [**DLinear**]: Are Transformers Effective for Time Series Forecasting?[\[AAAI 2023\]](https://arxiv.org/pdf/2205.13504.pdf)
[\[GitHub Repo\]](https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py)

5. [**FEDformer**]: FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting[\[ICML 2022\]](https://proceedings.mlr.press/v162/zhou22g.html)[\[GitHub Repo\]](https://github.com/thuml/Time-Series-Library/blob/main/models/FEDformer.py)

6. [**FreTS**]: Frequency-domain MLPs are More Effective Learners in Time Series Forecasting[\[NeurIPS 2023\]\(https://arxiv.org/pdf/2311.06184.pdf)[\[GitHub Repo\]](https://github.com/thuml/Time-Series-Library/blob/main/models/FreTS.py)
   
7. [**FiLM**]: FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting[\[NeurIPS 2022\]\(https://openreview.net/forum?id=zTQdHSQUQWc)[\[GitHub Repo\]](https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py)
   
8. [**FourierGNN**]: FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective[\[NeurIPS 2023\]\(https://arxiv.org/pdf/2311.06190.pdf)[\[GitHub Repo\]](https://github.com/aikunyi/FourierGNN)


🌟**Acknowledgement**

We appreciate the following GitHub repos for their valuable code and efforts.
- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- FourierGNN (https://github.com/aikunyi/FourierGNN)
