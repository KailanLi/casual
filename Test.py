#%%
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv(r'C:\Users\adaml\Python\Fairness Systems\causal.csv')






# %%
def preprocess_data(df):
    # Replace "?" with "unknown"
    df.replace("?", "unknown", inplace=True)

    # Remove blank spaces
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Change 'race' to binary values first
    non_numeric_columns = list(df.select_dtypes(exclude=[np.number]).columns)
    le = LabelEncoder()

    for col in non_numeric_columns:
        df[col] = le.fit_transform(df[col])
    print(non_numeric_columns)


    return df
df = preprocess_data(df)


# %% 
import warnings
from causalnex.structure import StructureModel

warnings.filterwarnings("ignore")  # silence warnings
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
sm = StructureModel()
from causalnex.structure.notears import from_pandas
sm = from_pandas(df)

viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
# %% 
viz.toggle_physics(False)
viz.show("01_fully_connected.html")


# %% 

sm.remove_edges_below_threshold(0.8)
viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK
)
viz.show("supporting_files/01_thresholded.html")























