import numpy as np
from torch import Tensor
import torch


class SequenceLabeling_Test():
	def __init__(self):

		# Sample input for Sequence Labelinb: 
		self.input_ids = torch.tensor([[  101,  1031, 13371,  5709,  1024,  2382, 13938,  2102,  1033,   102,
										 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
										 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
										 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
										 0,     0,     0,     0,     0,     0,     0,     0],
									[  101,  1996,  2231,  1011,  2805,  3422, 16168,  1010,  2529,  2916,
									  2120,  9970,  1010,  2001,  9339,  2023,  3204,  2011,  2334,  6399,
									  2004,  3038,  2055,  1015,  1010,  4278,  9272,  2031,  2351,  1999,
									  5968,  4491, 11248,  2006,  9587, 25016,  2213,  8431,  1999,  1996,
									  2627,  2048,  2086,  1012,   102,     0,     0,     0],
									[  101,  4715,  1011,  2413,  2223,  7680,  7849,  3111,  1012,   102,
										 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
										 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
										 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
										 0,     0,     0,     0,     0,     0,     0,     0]])
		self.masks = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
									 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
									[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
									 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
									[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
									 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

		self.token_type_ids = torch.zeros(self.masks.shape, dtype=torch.int64)

		self.labels = torch.tensor([[0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
									 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
									[0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
									 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
									[0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
									 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

		self.outputs = torch.tensor([[[-4.8311e-01,  3.9632e-03,  1.9482e-01,  3.3252e-02, -2.7929e-01,
							   6.3012e-02, -2.2961e-01, -1.4923e-01, -3.2148e-02],
							 [-9.8466e-02,  1.0897e-01,  5.2619e-01, -4.7399e-02,  5.3184e-02,
							  -2.8547e-01, -2.0070e-01, -7.0003e-01,  2.9308e-01],
							 [ 1.6138e-01, -3.0687e-01,  1.0181e-01, -9.0217e-02, -2.3237e-01,
							  -2.1505e-01,  5.0648e-02, -4.1570e-01,  1.8353e-01],
							 [ 3.3146e-01, -6.2549e-01, -3.9453e-01, -2.9985e-02, -9.8815e-01,
							  -6.6902e-02,  3.2248e-01, -3.0891e-01,  1.5018e-01],
							 [ 4.3927e-01, -3.3567e-01,  2.8348e-01,  6.4515e-02, -7.0393e-01,
							  -3.7512e-01,  4.9202e-01,  8.9874e-02, -5.7818e-03],
							 [ 3.9722e-01, -8.4037e-01,  1.7094e-01,  9.4486e-02, -5.2357e-01,
							   3.6890e-02,  5.8254e-01, -5.7305e-02,  4.2727e-01],
							 [ 3.7442e-01, -5.0841e-01, -1.5823e-01,  1.5119e-01, -5.8652e-01,
							  -1.9300e-01,  3.0373e-01, -3.4053e-01,  6.0673e-01],
							 [ 6.0206e-01, -2.4779e-01,  1.3428e-01,  2.8796e-01, -4.8979e-01,
							  -4.5788e-01,  9.8391e-02,  4.9989e-03,  4.5863e-01],
							 [ 1.8434e-01,  1.9019e-01,  5.3508e-01,  2.1845e-01,  1.1712e-01,
							   2.0472e-01, -4.2559e-02, -7.1926e-01,  3.7453e-01],
							 [-1.4779e-01, -6.0676e-01,  9.7696e-02, -2.2416e-01,  5.1533e-01,
							  -4.2705e-01, -1.1181e-01, -1.3843e-01,  2.5447e-01],
							 [-6.8929e-02,  3.8629e-01,  1.0102e-01,  5.4769e-02, -2.6051e-01,
							  -1.0486e-01, -1.5002e-01, -2.9452e-01,  2.6509e-01],
							 [-1.3317e-01,  4.4421e-01,  1.0418e-01,  1.5601e-01, -2.8926e-01,
							  -1.1145e-01, -2.3716e-01, -3.4433e-01,  4.3592e-01],
							 [-1.1879e-01,  3.6360e-01,  5.7713e-02, -2.1450e-02, -2.1675e-01,
							  -7.4655e-02, -1.6806e-01, -2.7648e-01,  9.0261e-02],
							 [-1.1976e-01,  4.7882e-01,  1.3649e-01,  1.8270e-01, -2.8884e-01,
							  -1.0904e-01, -2.7909e-01, -3.5578e-01,  4.5519e-01],
							 [-1.1599e-01,  4.2836e-01,  1.1915e-01,  7.5450e-02, -1.8366e-01,
							  -1.2322e-01, -2.5388e-01, -2.9894e-01,  2.6948e-01],
							 [-8.0065e-02,  2.4673e-01, -2.4461e-02, -8.9744e-02, -1.5050e-01,
							  -6.7603e-02, -2.3362e-01, -2.3542e-01, -1.5353e-01],
							 [-3.6435e-02,  1.7202e-01, -1.0848e-01, -3.4880e-02, -3.5342e-02,
							  -1.1680e-01, -1.8249e-01, -1.8084e-01, -1.3161e-01],
							 [-4.6907e-02,  2.5491e-01, -2.1197e-02, -8.1304e-02, -1.2009e-01,
							  -1.5220e-02, -2.0781e-01, -2.1432e-01, -1.2141e-01],
							 [-1.1639e-02,  2.5572e-01, -1.4557e-02, -6.3237e-02, -1.0144e-01,
							  -5.7845e-02, -1.8836e-01, -2.1252e-01, -1.2493e-02],
							 [ 6.0491e-02,  4.9945e-01,  2.1007e-01,  1.5786e-01, -1.8855e-01,
							  -7.9476e-02, -2.2501e-01, -3.4373e-01,  3.5213e-01],
							 [ 1.6137e-02, -4.0168e-03, -1.0859e-01, -2.7406e-01, -6.3735e-02,
							  -2.2137e-04, -1.6174e-01, -1.7320e-01, -1.7880e-01],
							 [-5.8286e-03,  1.9003e-01,  8.3676e-02, -6.8193e-02, -1.9010e-02,
							  -1.9521e-01, -1.5285e-02, -2.0516e-01,  1.5877e-01],
							 [ 5.2566e-02,  5.1794e-01,  1.7550e-01,  1.5497e-01, -1.5660e-01,
							  -2.0033e-01, -1.9730e-01, -3.5924e-01,  3.7791e-01],
							 [-8.9361e-02,  4.1665e-01,  9.2276e-02,  1.7327e-02, -1.2308e-01,
							  -1.7740e-01, -1.7831e-01, -2.9933e-01,  3.4094e-01],
							 [-5.2822e-02,  4.9349e-01,  1.5597e-01,  8.7760e-02, -2.4733e-01,
							  -1.2794e-01, -2.7611e-01, -3.3539e-01,  4.3566e-01],
							 [-9.2127e-02,  1.9364e-01,  8.7227e-02, -1.9258e-01, -2.7401e-02,
							  -2.2864e-01, -2.1738e-01, -2.2921e-01, -1.1045e-01],
							 [-8.5641e-02,  2.3147e-01,  2.7039e-02, -1.9060e-01, -1.5270e-01,
							  -3.2218e-01, -1.9850e-01, -1.3865e-01, -1.5546e-01],
							 [-4.4781e-02,  4.7989e-02, -1.2559e-01, -1.3249e-01, -1.9883e-02,
							  -3.7525e-01, -1.9945e-01, -8.9741e-02, -1.3277e-01],
							 [ 6.3546e-02,  6.0224e-02, -1.7255e-01, -1.7295e-01, -1.2057e-01,
							  -2.9637e-01, -2.7264e-01, -8.5885e-02, -2.4915e-01],
							 [ 3.8309e-02,  2.4464e-02, -2.2920e-01, -1.5887e-01, -1.9449e-02,
							  -3.4662e-01, -3.1578e-01, -9.5008e-02, -2.1295e-01],
							 [ 3.6290e-02,  1.0126e-02, -2.1578e-01, -1.9841e-01, -1.2632e-02,
							  -2.7522e-01, -2.7679e-01, -7.5876e-02, -2.5488e-01],
							 [-1.5649e-01, -3.8598e-01, -2.9810e-01, -2.4785e-01, -3.8666e-01,
							   1.9654e-01,  2.0553e-02, -3.9987e-01, -4.2337e-01],
							 [-1.0842e-01,  5.0165e-03, -1.8392e-01, -1.5132e-01, -2.6098e-01,
							  -1.2422e-01, -9.8091e-02, -1.3796e-01, -1.4839e-01],
							 [ 1.1347e-01,  5.1690e-01,  1.9349e-01,  1.4629e-01, -2.2322e-01,
							  -1.9917e-01, -6.5411e-02, -3.3491e-01,  3.3294e-01],
							 [ 7.9026e-02,  1.5809e-01,  6.4277e-02, -3.1010e-02, -9.6776e-02,
							  -2.5545e-01,  5.3856e-02, -1.8450e-01,  2.3447e-01],
							 [ 4.9282e-02,  5.4087e-01,  2.0166e-01,  1.5336e-01, -2.7717e-01,
							  -1.5696e-01, -1.8693e-01, -3.7132e-01,  4.0258e-01],
							 [-8.3586e-02,  2.3913e-01,  8.5976e-02, -9.1315e-02, -1.1059e-01,
							  -1.7554e-01, -6.0684e-02, -2.1129e-01,  1.3246e-01],
							 [-7.2410e-02,  3.1326e-01,  9.3124e-02, -3.5510e-02, -9.1141e-02,
							  -1.7683e-01, -5.0230e-02, -2.5221e-01,  1.8236e-01],
							 [-3.1854e-02,  5.0999e-01,  1.8602e-01,  1.4971e-01, -2.0711e-01,
							  -2.1234e-01, -2.1287e-01, -3.6135e-01,  4.4441e-01],
							 [-1.4662e-01,  4.3017e-01,  1.0234e-01,  4.6742e-02, -2.0685e-01,
							  -1.5604e-01, -2.0741e-01, -3.3104e-01,  3.6297e-01],
							 [-1.0349e-01,  3.0286e-01,  7.4926e-02, -3.1729e-02, -1.2337e-01,
							  -9.5593e-02, -2.1299e-01, -2.8392e-01,  1.3133e-01],
							 [-8.3359e-02,  3.0163e-01,  8.2026e-02, -8.9178e-02, -8.5283e-02,
							  -1.3802e-01, -1.9102e-01, -2.3272e-01,  1.1201e-02],
							 [-5.2524e-02,  2.9047e-01,  3.4780e-02, -6.7910e-02, -8.4446e-02,
							  -1.0695e-01, -2.1318e-01, -2.2488e-01, -3.0873e-02],
							 [ 1.1711e-02,  4.5738e-01,  1.3979e-01,  1.3191e-01, -2.2219e-01,
							  -9.7423e-02, -2.6113e-01, -2.9274e-01,  2.8748e-01],
							 [-2.4763e-03,  2.0686e-01, -2.5353e-02, -1.1769e-01, -1.0397e-01,
							  -8.7437e-02, -1.9353e-01, -1.9093e-01, -1.0155e-01],
							 [ 3.2828e-02,  1.1473e-01, -1.1302e-01, -9.5958e-02, -4.0885e-02,
							  -1.2071e-01, -2.0935e-01, -1.6174e-01, -1.2622e-01],
							 [ 3.1375e-02, -3.5074e-02, -1.5552e-01, -1.6823e-01,  5.5151e-02,
							  -3.5696e-02, -2.1742e-01, -1.7728e-01, -1.9645e-01],
							 [-5.7195e-02, -2.7544e-01, -2.9282e-01, -3.6411e-01, -2.8020e-01,
							   1.1803e-01, -9.3171e-02, -3.0051e-01, -1.7965e-01]],
							[[-3.1832e-01, -2.6598e-01,  2.3015e-03,  6.1641e-02, -5.1080e-01,
							   5.2985e-01,  1.7412e-01, -4.2558e-01, -8.3798e-02],
							 [-1.2666e-01, -1.6595e-02, -4.5265e-03,  1.6977e-01, -2.8749e-01,
							  -1.0539e-01,  2.2263e-01, -4.6749e-03, -1.4466e-01],
							 [ 8.4139e-02, -1.8751e-01,  2.3222e-01,  2.4501e-01, -4.8005e-01,
							  -2.3351e-01,  5.3955e-01, -6.8018e-01,  1.3775e-01],
							 [ 1.6539e-01, -1.8761e-01, -1.8837e-01,  3.3336e-01, -6.8282e-01,
							  -6.1802e-01,  7.0344e-01, -6.7934e-01,  1.8473e-01],
							 [-2.1691e-01, -3.6467e-01, -5.2835e-01,  1.8010e-01, -4.8147e-01,
							  -2.3734e-01,  7.1068e-01, -4.5338e-01,  7.7622e-02],
							 [ 1.0784e-01, -3.6518e-01, -3.4725e-01, -3.3509e-01, -3.7676e-02,
							   1.3087e-01,  4.9357e-01, -6.0246e-01,  3.6133e-01],
							 [-1.3074e-01, -4.3823e-01, -1.3980e-01, -3.5066e-01, -9.3367e-02,
							  -1.1183e-01,  8.6803e-02, -2.3333e-01, -5.7693e-02],
							 [-3.4807e-01, -5.7124e-01, -7.8313e-02, -2.1133e-03, -4.7648e-01,
							  -5.0438e-01,  7.9102e-02, -3.3781e-01, -1.6137e-01],
							 [ 5.6672e-02, -8.2959e-01,  2.6555e-02, -3.4968e-01,  1.8073e-01,
							  -1.9275e-01,  2.0463e-01, -5.4837e-01,  3.7573e-01],
							 [ 5.4598e-02, -5.2979e-01, -3.1297e-01, -4.3942e-01,  4.5000e-01,
							  -7.8634e-02, -5.7888e-01, -1.4397e-01,  1.5295e-02],
							 [-2.6119e-01, -7.7070e-02,  4.2784e-02, -1.6906e-01, -1.8249e-01,
							  -1.7672e-01,  2.0649e-01, -3.2973e-01,  5.2380e-02],
							 [-4.3802e-01, -1.6138e-01,  6.0373e-02, -9.1812e-02, -2.8721e-02,
							   3.9304e-01,  1.4028e-01, -3.4817e-01, -3.0853e-01],
							 [-3.0941e-01, -7.4108e-01,  8.5484e-02, -2.1276e-01,  4.4985e-01,
							  -2.1295e-01, -1.3549e-01, -2.2454e-01,  2.3859e-01],
							 [-1.6725e-01, -2.0416e-01,  2.5250e-01,  5.5772e-02, -1.9869e-01,
							  -3.2142e-01,  9.6924e-02, -5.6145e-01,  3.2070e-01],
							 [-3.5704e-02,  5.3146e-02,  6.8485e-02, -8.5429e-03, -2.1498e-01,
							  -4.8022e-01,  2.5049e-01, -5.4514e-01, -8.6357e-02],
							 [ 8.6576e-01, -1.9084e-02, -1.9754e-01, -8.5024e-02, -5.4089e-01,
							  -2.7381e-01,  4.1741e-01,  5.7453e-02,  5.2155e-01],
							 [ 1.4697e-01, -1.8176e-01, -1.5491e-01,  1.3682e-01, -3.5987e-01,
							  -2.1092e-01,  6.8843e-01, -4.5417e-01,  3.1900e-01],
							 [ 1.3751e-01, -3.9011e-01, -2.1929e-01,  6.2406e-01, -5.8802e-01,
							  -1.5951e-01, -1.0929e-01, -4.7717e-01, -3.5295e-01],
							 [ 8.8338e-02, -5.6367e-01, -8.1498e-03,  1.1911e-01, -3.5017e-01,
							   3.1483e-02,  6.4152e-01, -5.9647e-01,  3.6033e-02],
							 [-2.0768e-01,  2.1537e-02,  2.8512e-01,  5.7191e-01, -2.6066e-01,
							  -1.2237e-01, -1.4259e-02, -4.9900e-01,  2.4239e-02],
							 [-2.4196e-01,  3.3421e-02,  1.4492e-01, -2.7820e-01, -1.2111e-01,
							  -6.5041e-01,  1.1823e-01, -6.6713e-01,  5.7668e-02],
							 [ 2.2241e-01,  7.7698e-02,  1.0888e-01,  9.5569e-02, -1.1713e-01,
							  -2.0981e-01,  5.1039e-01, -5.9222e-01, -6.3967e-01],
							 [ 1.8173e-01, -4.0881e-01, -3.2173e-02,  2.8478e-01, -3.5642e-01,
							   2.5478e-01,  1.3449e-01, -4.5436e-01, -4.9933e-02],
							 [ 2.4054e-01, -2.7643e-01, -1.7345e-01,  4.3644e-01, -4.9544e-01,
							   5.2511e-02,  2.0866e-01, -5.8145e-01,  2.1400e-02],
							 [ 4.5646e-02, -7.2188e-01, -4.2111e-01,  3.0553e-01, -5.3650e-01,
							  -2.1924e-01,  2.6681e-01, -7.3100e-01,  1.6105e-01],
							 [ 6.3228e-02, -4.5957e-01, -2.2723e-01, -2.0615e-01, -2.1606e-01,
							   1.9995e-01,  3.3946e-01, -4.3961e-01, -2.4148e-01],
							 [-1.9871e-01, -2.6340e-01, -3.5241e-01, -2.2521e-02, -1.4373e-01,
							   1.0303e-01,  2.8715e-01, -7.5317e-01, -2.9140e-02],
							 [-2.2485e-01, -6.2774e-02,  2.0464e-01,  1.8131e-01, -4.8979e-01,
							  -5.1081e-01,  5.1188e-01, -5.5940e-01, -1.7522e-01],
							 [-2.4885e-01,  4.4268e-02, -2.1414e-01, -1.9211e-01, -1.8919e-01,
							  -4.6680e-02,  3.1887e-01, -3.7929e-01, -5.1259e-01],
							 [-1.0434e-01, -7.6913e-02, -1.0275e-02, -1.7845e-02, -2.4663e-01,
							   9.2087e-02,  2.0631e-01, -2.9215e-01, -4.9443e-01],
							 [ 8.1677e-02, -4.0988e-01, -3.8007e-01,  4.9877e-01, -1.6050e-01,
							  -3.2167e-02,  5.5222e-01, -6.4178e-01,  1.0379e-01],
							 [-1.3732e-01, -2.0731e-01, -4.9252e-01,  1.4517e-01, -4.9750e-03,
							  -2.6278e-02,  1.0834e-01, -2.7032e-01, -2.0136e-01],
							 [-1.0691e-01,  8.3071e-03, -5.8713e-01, -1.9683e-01, -1.3326e-01,
							  -6.1112e-02,  3.5663e-01, -4.5524e-01,  5.0322e-02],
							 [-1.2615e-01, -1.7027e-01,  2.4322e-02, -6.7986e-02, -3.8408e-01,
							  -4.3487e-01,  3.6310e-01, -4.4920e-01,  1.5431e-01],
							 [ 3.7864e-02, -5.1956e-01, -1.0276e-01,  4.0241e-01, -3.9713e-01,
							  -4.0072e-01,  5.4306e-01, -6.5177e-01,  5.3252e-01],
							 [ 4.3553e-01, -7.3344e-01,  3.8405e-02,  3.4372e-01, -6.0916e-01,
							   8.3549e-02,  2.9896e-01, -3.2571e-01,  7.3960e-01],
							 [ 1.1227e-01, -7.0183e-01, -3.5174e-01,  1.0813e-01, -5.1910e-01,
							  -1.1181e-01,  3.6989e-01, -5.5167e-01,  4.6100e-01],
							 [-1.6495e-02, -5.1331e-01, -4.0408e-01,  2.2058e-01, -4.4482e-01,
							  -2.5640e-01,  3.1984e-01, -6.5229e-01,  2.8994e-01],
							 [ 4.3008e-02, -6.8891e-01, -1.9242e-01,  1.2941e-01, -1.3272e-01,
							   1.8295e-01,  6.5480e-02, -8.6759e-02,  1.4995e-01],
							 [ 5.9415e-01, -9.2023e-01, -2.7121e-01,  1.0769e-01, -3.2145e-01,
							   8.8809e-02,  6.8021e-01,  6.9548e-02,  4.4884e-02],
							 [ 1.7845e-01, -5.1827e-01, -4.0091e-01,  3.2517e-01, -3.8902e-01,
							   9.9345e-02,  5.0982e-01, -3.3879e-01,  7.1454e-01],
							 [ 7.1089e-02, -8.1838e-01, -5.0532e-02,  1.6048e-01,  1.3809e-01,
							   5.4605e-03,  3.8414e-01, -3.4210e-04,  3.9860e-01],
							 [ 2.0065e-01, -4.3899e-01, -2.4727e-01,  6.0972e-02, -5.7625e-01,
							  -3.5542e-01,  5.3445e-01, -4.7636e-01,  4.4243e-01],
							 [-2.5617e-01, -5.0448e-01,  1.3482e-01, -2.1904e-01,  2.6850e-01,
							  -1.2347e-01, -1.3837e-01, -1.2200e-01,  3.1633e-01],
							 [ 2.6507e-01, -6.5620e-01,  2.2757e-02,  1.6272e-01, -3.8237e-01,
							  -2.4421e-01,  8.5053e-01, -5.0112e-01,  3.4773e-01],
							 [ 6.3616e-02,  2.3969e-01, -1.8467e-01,  6.5965e-02, -1.1205e-01,
							  -4.9231e-02,  6.8260e-03, -2.7340e-01, -9.1009e-02],
							 [ 2.5581e-02,  2.9961e-01, -1.9244e-01,  1.2233e-01, -1.5188e-01,
							  -4.6692e-02,  8.0743e-02, -3.1022e-01,  6.4190e-02],
							 [ 3.1754e-03,  1.3749e-02, -1.8837e-01,  3.1287e-02, -1.7338e-02,
							   2.4716e-02, -7.6512e-02, -3.1423e-01, -2.8314e-01]],
							[[-3.7075e-01, -4.0878e-01,  4.5449e-01,  3.1278e-01, -4.6542e-02,
							   2.6541e-01,  1.9744e-01, -1.4241e-01, -1.4079e-01],
							 [-2.8991e-01,  3.9399e-01,  1.0592e-01,  7.0179e-04,  7.1630e-02,
							   2.4421e-01,  3.5057e-01,  3.6483e-02, -1.2571e-01],
							 [-9.2202e-02,  1.5987e-02,  3.5998e-02,  2.3962e-01, -5.3795e-02,
							  -1.3172e-01, -1.0349e-01, -2.7279e-01,  5.5340e-02],
							 [ 7.7313e-02, -3.1164e-01,  6.2081e-02,  7.5457e-02, -5.9561e-02,
							   6.7634e-01,  3.0119e-03, -7.8542e-02,  2.0488e-02],
							 [ 2.1949e-01, -1.8383e-01, -3.7173e-01,  1.3551e-01,  9.8958e-02,
							   6.9336e-01,  2.6192e-01, -3.2222e-02,  1.4482e-01],
							 [-1.3873e-01, -8.0589e-02,  3.8372e-01,  8.4027e-02,  1.0734e-03,
							  -1.6457e-01,  2.3710e-02, -4.6153e-01,  1.0398e-01],
							 [-1.9795e-01, -3.4985e-01,  1.8907e-01, -2.4551e-01,  3.4133e-02,
							  -2.3259e-01,  1.1219e-01, -5.8695e-01,  2.1117e-01],
							 [-2.5242e-02, -3.9103e-01,  2.5498e-01, -2.7228e-02, -1.2145e-01,
							   3.9772e-02, -2.2114e-01, -5.1512e-01,  8.9405e-02],
							 [-2.2895e-01,  6.7355e-02,  2.1167e-01,  2.1705e-01, -9.4111e-02,
							  -1.7273e-01, -2.0252e-01, -5.3279e-01,  1.5203e-01],
							 [-1.0557e-01, -5.8617e-01,  2.5082e-01, -9.3544e-02,  5.4118e-01,
							  -3.5988e-01, -9.6325e-02, -6.1940e-02,  1.8624e-01],
							 [-1.5268e-02,  4.9114e-02,  4.7386e-02,  1.4595e-01, -5.1168e-02,
							  -1.1638e-02, -1.8382e-01, -1.6205e-01, -1.7480e-01],
							 [ 4.6526e-02,  1.5260e-01,  1.2980e-01,  2.4006e-01, -5.4803e-02,
							  -1.1411e-02, -1.8286e-01, -2.2315e-01, -7.5693e-02],
							 [-9.5331e-02,  1.0272e-01, -1.3345e-02,  6.4892e-02,  8.2968e-02,
							   6.0453e-02, -2.3009e-01, -1.6819e-01, -2.2404e-01],
							 [ 3.4642e-02,  2.5368e-01,  1.9847e-01,  3.5358e-01, -7.5023e-02,
							   1.4247e-02, -1.9931e-01, -3.0829e-01,  4.6079e-02],
							 [ 1.5002e-02,  5.8394e-02,  6.7255e-02,  2.1412e-01,  1.0681e-02,
							   5.7739e-04, -2.8326e-01, -1.7053e-01, -1.2128e-01],
							 [-5.6338e-02,  1.0368e-01, -6.4199e-03,  1.6401e-02,  1.4416e-01,
							   1.1046e-01, -2.2571e-01, -1.5514e-01, -2.1867e-01],
							 [-5.4808e-02,  2.5345e-02,  4.4264e-02,  1.4115e-01,  3.2337e-02,
							  -1.8059e-02, -2.7310e-01, -1.9471e-01, -1.8436e-01],
							 [-9.5595e-02,  1.4585e-01, -2.7053e-02,  1.0372e-01,  7.3944e-02,
							   1.0662e-01, -2.6141e-01, -1.7764e-01, -1.5934e-01],
							 [-8.9963e-02,  1.3674e-01, -1.6151e-02,  1.6347e-01,  1.0152e-01,
							   6.3958e-02, -1.5818e-01, -1.6732e-01, -1.0971e-01],
							 [-6.1990e-02,  3.1198e-01,  2.1938e-01,  4.1123e-01,  4.0202e-03,
							  -1.7046e-01, -1.7129e-01, -4.3517e-01,  1.6986e-01],
							 [-1.7967e-01,  7.2879e-02, -1.2676e-01, -2.5752e-02,  3.3389e-02,
							   1.1490e-01, -1.3767e-01, -1.7412e-01, -2.8926e-01],
							 [-1.9690e-01,  1.3013e-01, -7.5128e-02,  2.6257e-03, -1.6337e-02,
							  -1.6231e-02, -3.2230e-01, -2.6191e-01, -2.2871e-01],
							 [ 2.3475e-02,  3.1413e-01,  2.2357e-01,  3.9859e-01, -6.8235e-02,
							  -5.7757e-02, -2.4281e-01, -4.1192e-01,  1.3835e-01],
							 [-1.1864e-01,  1.1373e-01, -5.1464e-02,  6.7025e-02,  5.7053e-02,
							   1.6893e-02, -2.4461e-01, -1.8492e-01, -1.7494e-01],
							 [ 7.4076e-02,  2.2649e-01,  2.3322e-01,  3.0672e-01, -4.2358e-02,
							   4.0600e-02, -2.1021e-01, -2.8658e-01,  3.8389e-02],
							 [-9.0774e-03, -7.2843e-03,  4.2573e-02,  6.0527e-02, -1.5505e-02,
							   6.2596e-02, -2.5404e-01, -5.9665e-02, -2.5996e-01],
							 [ 4.9523e-03,  1.1825e-01, -2.4946e-02, -5.1887e-02,  2.6229e-01,
							   2.1631e-01, -1.3003e-01, -2.8290e-01, -2.1744e-01],
							 [-7.2865e-02,  1.0085e-02,  7.6249e-02,  1.2848e-01,  1.5170e-02,
							   5.6298e-03, -2.6844e-01, -1.7672e-01, -2.3137e-01],
							 [-8.8901e-02,  1.3666e-02,  4.4560e-02,  5.9482e-02, -8.6346e-02,
							   9.1576e-02, -2.9515e-01, -3.7088e-02, -3.2740e-01],
							 [-2.3080e-01,  8.5368e-02, -5.0778e-02,  1.6647e-01,  3.2722e-02,
							  -1.0363e-02, -7.2057e-02, -1.6618e-01, -1.9148e-01],
							 [-2.4918e-01,  7.4442e-02, -4.8167e-02,  8.6732e-02, -3.9936e-02,
							   2.0132e-02, -2.9346e-03, -1.1114e-01, -2.1781e-01],
							 [-3.2161e-01,  7.6802e-02, -4.0256e-02, -1.1360e-02,  8.4057e-03,
							  -1.3090e-03, -2.1243e-03, -1.0247e-01, -1.9101e-01],
							 [-2.7274e-01,  4.6553e-02, -3.5267e-02, -8.8885e-02,  2.7700e-02,
							  -5.1016e-02, -2.3690e-01, -3.1523e-01, -2.3792e-01],
							 [-2.1008e-01,  2.2121e-01,  3.4474e-02,  2.4993e-01, -3.2401e-02,
							  -2.2897e-02, -1.0380e-01, -2.5968e-01,  1.5713e-02],
							 [-2.4911e-01,  6.0407e-02, -5.9990e-02, -6.2915e-02, -4.8457e-02,
							   2.9854e-02, -3.0831e-01, -3.8317e-01, -2.7649e-01],
							 [-6.6963e-02,  2.8865e-01,  1.6326e-01,  3.5200e-01, -1.4500e-01,
							  -2.7961e-02, -1.9418e-01, -3.0105e-01,  5.0064e-02],
							 [-2.7741e-01,  1.1900e-01, -1.2913e-01,  8.4089e-02, -1.1329e-01,
							  -8.2951e-02, -2.9602e-01, -2.1158e-01, -2.5069e-01],
							 [-2.4931e-01,  1.2534e-01, -1.1743e-01,  9.8849e-02, -1.1497e-01,
							  -8.9732e-02, -2.8368e-01, -1.9874e-01, -2.5843e-01],
							 [ 1.1652e-01,  2.9057e-01,  2.3342e-01,  3.7582e-01, -1.7245e-01,
							   1.9339e-02, -1.2939e-01, -4.0824e-01,  7.4953e-02],
							 [ 6.4463e-02,  4.2078e-02,  6.3090e-02,  2.0199e-01, -3.3053e-03,
							  -1.1582e-03, -2.4519e-01, -2.0000e-01, -1.2395e-01],
							 [ 6.3834e-02,  5.9522e-02,  9.9747e-02,  2.3283e-01,  2.9581e-02,
							  -2.9340e-03, -2.4062e-01, -1.8083e-01, -1.0326e-01],
							 [ 1.9408e-02, -2.0842e-02,  2.7745e-02,  1.3954e-01,  2.5216e-02,
							  -1.2431e-02, -3.1182e-01, -1.6704e-01, -1.9468e-01],
							 [-4.9630e-02,  1.1032e-01,  2.7874e-03,  2.4921e-02,  1.7280e-01,
							   6.8318e-02, -2.7298e-01, -1.7779e-01, -1.8979e-01],
							 [-4.6984e-02,  3.0413e-01,  1.6464e-01,  3.8118e-01,  1.4103e-02,
							  -8.3913e-02, -2.2913e-01, -2.3332e-01,  6.7047e-02],
							 [ 1.3061e-02,  1.8325e-02, -4.1086e-03,  1.5526e-01,  4.6003e-02,
							   5.9591e-02, -2.3612e-01, -1.7848e-01, -1.7394e-01],
							 [-3.3048e-02,  4.9637e-02, -1.9849e-02,  1.3452e-01,  7.1716e-02,
							   6.6603e-02, -6.3773e-02, -1.4937e-01, -1.6911e-01],
							 [-3.0349e-01, -9.0695e-02, -6.2612e-02, -1.2194e-01, -2.8283e-01,
							   4.5493e-02, -1.1113e-01, -2.9803e-01, -2.6693e-01],
							 [-1.5222e-01, -4.5105e-02, -2.6307e-01, -3.3720e-01,  1.5553e-01,
							  -5.4994e-02, -2.7401e-01, -1.7495e-01, -1.3675e-01]]])
		