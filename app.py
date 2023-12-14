import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Titanic �����ͼ��� �ҷ����� (����: CSV ���� ���·� ����Ǿ� �ִٰ� ����)
# ���� ��θ� ���� ������ ���� ��η� �ٲ��ּ���.
titanic_data = pd.read_csv('./data/titanic.csv')

# ���̰� ����ġ�� �� ����
titanic_data = titanic_data.dropna(subset=['Age'])

# Pclass���� �׷�ȭ�ϰ� ������ �� ���
survival_by_pclass_age = titanic_data.groupby(['Pclass', pd.cut(titanic_data['Age'], bins=range(0, 81, 10))])['Survived'].mean().unstack()

# Streamlit ���ø����̼� ����
st.title('Survival Rate by Age and Pclass')

# Matplotlib �׷����� Streamlit�� ����
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(survival_by_pclass_age, annot=True, fmt=".2%", cmap='Blues', cbar_kws={'label': 'Survival Rate'})
plt.title('Survival Rate by Age and Pclass')
plt.xlabel('Age Group')
plt.ylabel('Pclass')
plt.xticks(rotation=45)  # X�� �� ����̱�
st.pyplot(fig)

# Streamlit ���ø����̼� ����
if __name__ == '__main__':
    st.write("To view this Streamlit app, run the following command in your terminal:")
    st.code("streamlit run app.py")


