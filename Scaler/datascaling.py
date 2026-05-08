import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from DataPreprocessing.Sampler import DataSampling


class DataScaler:

    """

    ClassName  : DataScaler
    Description: This class is used to scale down the data between +1 to -1 using StandardScaler.
    Written By : Manthan Takalkar
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = DataSampling()

    def scale(self):

        """

        Method Name : scale
        Description : This method is used to scale down the independent data.
        Output      : scaled data(x)
        On_Failure  : Raise Exception

        Written By  : Manthan Takalkar
        Version     : 0.1
        Revisions   : None

        """

        try:
            x, y = self.data.sampling()
            scalar = StandardScaler()
            x = pd.DataFrame(scalar.fit_transform(x), columns=x.columns)
            print(x)
            return scalar, x, y
        except Exception as e:
            raise e

    def serializescalar(self):

        """

        Method Name : serializescalar
        Description : This method is used to save the scalar in the serialized format in the pickle file.
        Output      : Pickle file
        On_Failure  : Raise Exception

        Written By  : Manthan Takalkar
        Version     : 0.1
        Revisions   : None

        """

        try:
            import os
            scalar, x, y = self.scale()
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, 'Scalar.pkl')

            with open(file_path, 'wb') as file:
                pickle.dump(scalar, file)

            print("Scalar serialized and saved successfully!")
            
        except Exception as e:
            raise e


s = DataScaler()
s.serializescalar()