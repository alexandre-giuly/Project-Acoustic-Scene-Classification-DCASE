from .cnns import Cnn6_60k, Cnn6_60k_KD

nets = [Cnn6_60k_KD, Cnn6_60k]
#nets = [Cnn6_60k]
get_net = {str(n.__name__): n for n in nets}
