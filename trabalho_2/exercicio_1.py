from rule_generation import * 

# Carregando a base de dados de interesse.

df = pd.read_csv('s_temp_ubatuba.csv')
df.drop(columns='time', inplace=True)

# Transform the dataframe into lagged variables and target
df = df[['value']].copy()
df.columns = ['target']
for i in range(1, 12):
    df[f'lag_{i}'] = df['target'].shift(i)
df.dropna(inplace=True)

df.head()

# Transform the first column to last and so on
df = df.iloc[:, ::-1]
df.head()

min_value = min(df.min())
max_value = max(df.max())

variable_list = [ctrl.Antecedent(np.arange(min_value - 1, max_value + 1, 1), name) if name != 'target' else ctrl.Consequent(np.arange(min_value - 1, max_value + 1, 1), name) for name in df.columns]
for variable in variable_list:
    variable.automf(7)

fuzzy_system = create_fuzzy_system(df, variable_list, 1)

fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_system)

df['predict'] = np.nan
# Create a loop to predict the values 
for i in range(len(df)):
    l = df.iloc[i,:].shape[0] - 2
    for k in range(l):
        print(f'lag_{l - k}', l)
        fuzzy_sim.input[f'lag_{l - k}'] = df.iloc[i, k]
    fuzzy_sim.compute()

    df.iloc[i, l + 1] = fuzzy_sim.output['target']
# %%
import matplotlib.pyplot as plt 

plt.plot(df['target'], label='target')
plt.plot(df['predict'], label='predict')
# %%
df
# %%
