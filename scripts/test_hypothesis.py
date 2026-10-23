import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
products = pd.read_csv(os.path.join(data_dir, "products.csv"))

train['submitted_dt'] = pd.to_datetime(train['submitted_at'])
partners['onboarded_dt'] = pd.to_datetime(partners['onboarded_date'])

df = train.dropna(subset=['is_fraud']).copy()
df = df.merge(partners, on='partner_id', how='left').merge(products, on='sku', how='left')

df['tenure_days'] = (df['submitted_dt'] - df['onboarded_dt']).dt.days
df['tenure_months'] = df['tenure_days'] / 30.5
df['is_new_partner_6m'] = (df['tenure_days'] < 180).astype(int)
df['is_new_partner_1y'] = (df['tenure_days'] < 365).astype(int)
df['post_may'] = (df['submitted_dt'] >= '2026-05-01').astype(int)
df['under_2000'] = (df['claim_amount_inr'] < 2000).astype(int)

print("=== RAW FRAUD RATE BY PARTNER TENURE ===")
print("Tenure < 6m fraud rate:")
print(df.groupby('is_new_partner_6m')['is_fraud'].agg(['count', 'sum', 'mean']))

print("\nTenure < 1y fraud rate:")
print(df.groupby('is_new_partner_1y')['is_fraud'].agg(['count', 'sum', 'mean']))

# Bin tenure into quintiles
df['tenure_quintile'] = pd.qcut(df['tenure_days'], q=5)
print("\nTenure quintiles fraud rate:")
print(df.groupby('tenure_quintile')['is_fraud'].agg(['count', 'sum', 'mean']))

# Chi-squared and Fisher test for is_new_partner_6m
ct_6m = pd.crosstab(df['is_new_partner_6m'], df['is_fraud'])
chi2_6m, p_6m, _, _ = stats.chi2_contingency(ct_6m)
print(f"\nChi2 test for 6m tenure: chi2 = {chi2_6m:.4f}, p-value = {p_6m:.4e}")

# Check fraud concentration across partners
partner_fraud = df.groupby('partner_id').agg(
    total_claims=('claim_id', 'count'),
    fraud_claims=('is_fraud', 'sum'),
    fraud_rate=('is_fraud', 'mean'),
    tenure_mean=('tenure_days', 'mean'),
    city=('city', 'first'),
    partner_type=('partner_type', 'first')
).sort_values('fraud_claims', ascending=False)

print("\nTop 15 partners with most fraud claims:")
print(partner_fraud.head(15))
print(f"Total partners with at least 1 fraud claim: {(partner_fraud['fraud_claims'] > 0).sum()} out of {len(partner_fraud)}")

# Check if fraud is in specific partners regardless of tenure
print("\nTenure of top fraud partners:")
print(partner_fraud[partner_fraud['fraud_claims'] >= 3][['total_claims', 'fraud_claims', 'fraud_rate', 'tenure_mean', 'city']])

# Bootstrap CI for difference in fraud rates
np.random.seed(42)
diffs_6m = []
diffs_1y = []
for _ in range(1000):
    sample = df.sample(frac=1.0, replace=True)
    r_new_6m = sample[sample['is_new_partner_6m'] == 1]['is_fraud'].mean()
    r_old_6m = sample[sample['is_new_partner_6m'] == 0]['is_fraud'].mean()
    diffs_6m.append(r_new_6m - r_old_6m)
    
    r_new_1y = sample[sample['is_new_partner_1y'] == 1]['is_fraud'].mean()
    r_old_1y = sample[sample['is_new_partner_1y'] == 0]['is_fraud'].mean()
    diffs_1y.append(r_new_1y - r_old_1y)

ci_low_6m, ci_high_6m = np.percentile(diffs_6m, [2.5, 97.5])
ci_low_1y, ci_high_1y = np.percentile(diffs_1y, [2.5, 97.5])
print(f"\nBootstrap 95% CI for (Fraud_Rate_New - Fraud_Rate_Old) [6m]: {np.mean(diffs_6m):.4f} [{ci_low_6m:.4f}, {ci_high_6m:.4f}]")
print(f"Bootstrap 95% CI for (Fraud_Rate_New - Fraud_Rate_Old) [1y]: {np.mean(diffs_1y):.4f} [{ci_low_1y:.4f}, {ci_high_1y:.4f}]")

# Multivariate analysis using unpenalized / lightly penalized logistic regression
features = ['tenure_months', 'claim_amount_inr', 'customer_prior_claims', 'post_may', 'family', 'partner_type']
X = df[features].copy()
X['claim_amount_log'] = np.log(X['claim_amount_inr'])
X = X.drop(columns=['claim_amount_inr'])
y = df['is_fraud'].values

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['tenure_months', 'claim_amount_log', 'customer_prior_claims', 'post_may']),
        ('cat', OneHotEncoder(drop='first'), ['family', 'partner_type'])
    ]
)

pipe = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(C=1e5, max_iter=1000))])
pipe.fit(X, y)

feature_names = (['tenure_months', 'claim_amount_log', 'customer_prior_claims', 'post_may'] + 
                 list(pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(['family', 'partner_type'])))
coefs = pipe.named_steps['clf'].coef_[0]

print("\n=== MULTIVARIATE LOGISTIC REGRESSION COEFFICIENTS ===")
for name, coef in zip(feature_names, coefs):
    print(f"{name:35s}: {coef:+.4f} (odds ratio: {np.exp(coef):.4f})")
