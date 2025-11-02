def create_bar_chart(data, x_col, y_col, title="Bar Chart"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def create_line_chart(data, x_col, y_col, title="Line Chart"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], marker='o')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.tight_layout()
    return plt

def create_pie_chart(data, column, title="Pie Chart"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    data[column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(title)
    plt.ylabel('')
    plt.tight_layout()
    return plt