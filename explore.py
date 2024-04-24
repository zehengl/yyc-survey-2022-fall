# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from slugify import slugify
from tqdm import tqdm

meta_url = "https://data.calgary.ca/api/views/5geu-ar8w/files/09da28d9-0a00-44b8-b132-e879791f4f0d?download=true&filename=New%20Fall%20Variable%20Meta-data%202022%20Onward.xlsx"
data_url = "https://data.calgary.ca/resource/5geu-ar8w.json?$limit=3000"

df = pd.read_json(data_url)
meta = pd.read_excel(meta_url)

# %%
meta["variable"] = meta["variable"].str.lower()
meta["value"] = meta["value"].astype("Int64")
col_mapping = meta.groupby("variable").apply(lambda g: g["question_label"].iloc[0])
value_label_mapping = meta.groupby("variable").apply(
    lambda x: dict([(row["value"], row["valuelabel"]) for row in x.to_dict("records")])
)

# %%
cols = [col for col in df.columns if col in value_label_mapping]
for col in tqdm(cols):
    df[col] = df[col].apply(lambda val: value_label_mapping[col].get(val, val))


# %%
duplicates = col_mapping.value_counts()[col_mapping.value_counts() > 1].index.to_list()
col_mapping = col_mapping[col_mapping.apply(lambda v: v not in duplicates)]
mapping = {v: k for k, v in col_mapping.items()}

# %%
df = df.rename(columns=col_mapping.to_dict())
df

# %%
cols = [
    "Calgary is on the right track to be a better city 10 years from now",
    "How safe or unsafe do you think Calgary is overall?",
    "I am proud to be a Calgarian",
]
for col in tqdm(cols):
    ax = sns.countplot(
        data=df,
        x=col,
        order=value_label_mapping[mapping[col]].values(),
    )
    ax.tick_params(
        axis="x",
        labelsize=8,
        labelrotation=30,
    )
    ax.get_figure().savefig(
        f"docs/{slugify(col)}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

# %%
