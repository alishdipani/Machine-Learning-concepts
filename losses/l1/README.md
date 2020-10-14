## Formula

$$
L = \{l_{1}, l_{2}, ... l_{N}\}; \ \ \ \ \ \ l_{n} = |y^{pred}_{n} - y^{true}_{n}| \\
L1Loss(y^{pred}, y^{true}) = f_{reduce}(L); \ \ \ \ \ \ f_{reduce}\ \epsilon\ \{sum;\ mean;\ None \}
$$




## Standard Implementations in ML Frameworks
- [PyTorch: L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)
- [Keras: MeanAbsoluteError](https://keras.io/api/losses/regression_losses/#meanabsoluteerror-class)
- [TF 2.3: MAE](https://www.tensorflow.org/api_docs/python/tf/keras/losses/MAE)

## Other Resources
- [Analytics Vidhya Blog](https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/)
- [L1 vs. L2 Loss function](http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/)
- [Differences between L1 and L2 as Loss Function and Regularization](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)
