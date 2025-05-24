import Baseline
import Preprocessing

data = Preprocessing.load_data('Kumar.csv')
Baseline.multi_random_forest(data)