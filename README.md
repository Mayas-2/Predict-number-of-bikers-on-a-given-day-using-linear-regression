from IPython.display import clear_output
# Don't modify this code


%pip install gdown==4.5


clear_output()
# Download the CSV file.
!gdown 1_eJU8Y-31_l0oq1sSJT6pROJyo-ufuvD
import pandas as pd
import numpy as np
data_df = pd.read_csv('bikers_data.csv')
data_df.head()
data_y = data_df['Number of bikers'] # target
data_x = data_df.drop(['Number of bikers'], axis=1) # input features
data_x.head()
data_y
data_x['Month'] = data_x['Date'].apply(lambda x: int(x[5:7]))
data_x.head()
data_x.info()
data_x.drop('Date', inplace=True, axis=1)
data_x.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() # Feature Scaling

columns = data_x.columns
data_x = data_x.to_numpy()
data_x = scaler.fit_transform(data_x)
data_x = pd.DataFrame(data_x, columns=columns)
data_x.head()
data_y = scaler.fit_transform(np.array(data_y).reshape(-1, 1)) # MinMax Scaling
X = data_x.values
Y = data_y.squeeze()

X = np.hstack([X, np.ones((X.shape[0], 1))])
display(pd.DataFrame(X))
import sklearn
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

display(pd.DataFrame(x_train))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression(fit_intercept=False).fit(x_train, y_train)
ols_mse = mean_squared_error(y_test, lr.predict(x_test))

print(f'OLS MSE: {ols_mse}')
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas = [1e-2, 1e-1, 1, 1e1, 1e2], fit_intercept=False, store_cv_values=True)
ridge.fit(x_train, y_train)
y_ridge_pred = ridge.predict(x_test)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)
print(f'Cross Validation MSEs: {np.mean(ridge.cv_values_, axis=0)}')
print(f'Ridge MSE: {ridge_mse}, Alpha: {ridge.alpha_}')
from sklearn.linear_model import LassoCV

lasso = LassoCV(alphas = [1e-2, 1e-1, 1, 1e1, 1e2], fit_intercept=False, max_iter=100000)
lasso.fit(x_train, y_train)
y_lasso_pred = lasso.predict(x_test)
lasso_mse = mean_squared_error(y_test, y_lasso_pred)
print(f'LASSO MSE: {lasso_mse}, Alpha: {lasso.alpha_}')
print(f'OLS:\tMSE = {ols_mse}')
print(f'Ridge:\tMSE = {ridge_mse}')
print(f'LASSO:\tMSE = {lasso_mse}')
print(f'Ridge MSE: {ridge_mse}, Alpha: {ridge.alpha_}')
