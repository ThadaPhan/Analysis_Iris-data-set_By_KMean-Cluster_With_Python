import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('matplotlib', 'notebook')


def getTypeNames(l):
    typeNames = list()
    for i in range(1, l + 1):
        name = 'Cụm ' + str(i)
        typeNames.append(name)
    return np.array(typeNames)


def showCenters(centers, columns_):
    print('Trung tâm của mỗi cụm')
    index_ = getTypeNames(len(centers))
    ct = pd.DataFrame(centers, index=index_, columns=columns_)
    print(ct)
    print("\n\n\n")


def showCharts(data, labels, centers):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), dpi=120)
    fig.set_label('Sự tương quan cơ bản của dữ liệu')
    colors = ['r', 'b', 'k', 'y', 'm', 'g', 'c']
    for i in range(0, len(centers)):
        type_ = data.iloc[np.where(labels == i)[0]]
        color = np.random.choice(colors)
        colors.remove(color)

        ax1.plot(type_.iloc[:, 0], type_.iloc[:, 1], color + '+', label="Cụm " + str(i + 1))
        ax2.plot(type_.iloc[:, 2], type_.iloc[:, 3], color + '+', label="Cụm " + str(i + 1))

        ax1.plot(centers[i, 0], centers[i, 1], color + 'o', label="Trung tâm cụm " + str(i + 1))
        ax2.plot(centers[i, 2], centers[i, 3], color + 'o', label="Trung tâm cụm " + str(i + 1))

        ax1.legend(loc="lower left")
        ax2.legend(loc="upper left")
    ax1.set_title('Sepal');
    ax2.set_title('Petal')

    ax1.set_xlabel('Chiều dài');
    ax2.set_xlabel('Chiều dài')
    ax1.set_ylabel('Chiều rộng');
    ax2.set_ylabel('Chiều rộng')

    ax1.set_xlim(0, data['sepal.length'].max() + 1);
    ax2.set_xlim(0, data['petal.length'].max() + 1)
    ax1.set_ylim(0, data['sepal.width'].max() + 1);
    ax2.set_ylim(0, data['petal.width'].max() + 1)

    plt.legend()
    fig.show()
    print("\n\n\n")


def showElbowOfDataToChooseCluster(data):
    inertias = list()
    fig, _ = plt.subplots(1, 1, figsize=(8, 5), dpi=120)
    fig.set_label('Xác định số cụm để nhóm')
    for i in range(1, 10):
        kmeans = KMeans(n_clusters=i, random_state=1).fit(data.iloc[:, :4])
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, 10), inertias, 'rx-')
    plt.title('Tổng phương sai của mỗi cách chia cụm')
    plt.xlabel('Tổng số cụm')
    plt.ylabel('Tổng phương sai')
    fig.show()
    print('Vậy nên chia 3 chụm')
    print("\n\n\n")


def showBegin(data, labels, centers):
    colors = ['b', 'k', 'y', 'm', 'g', 'c']
    fig, _ = plt.subplots(1, 1, figsize=(8, 5), dpi=120)
    fig.set_label('Ban đầu')
    ax = plt.subplot(111, projection='3d')
    # img = ax.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=data.iloc[:,3],cmap=plt.hot())
    # c=fig.colorbar(img)
    # c.set_label('Chiều rộng đài hoa')
    mk = ['X', '^', 'D', '*', 'o']
    for i in range(len(centers)):
        type_ = data.iloc[np.where(labels == i)[0]]
        ax.scatter(type_.iloc[:, 0], type_.iloc[:, 1], type_.iloc[:, 2], c='b', s=type_.iloc[:, 3] * 50)
        ax.scatter(type_.iloc[:, 0], type_.iloc[:, 1], type_.iloc[:, 2], c='k', s=0.5)
    note = ax.scatter([centers[1, 0]], [centers[1, 1]], [centers[1, 2]], color='w', s=0)
    note.set_label('Đường kính của mỗi điểm là chiều rộng của đài hoa')
    fig.legend(loc="upper right")
    ax.set_title('Biểu đồ dữ liệu trước khi chia')
    ax.set_xlabel('Chiều dài cánh hoa')
    ax.set_ylabel('Chiều rộng cánh hoa');
    ax.set_zlabel('Chiều dài đài hoa');
    fig.show()


def showFinal(data, labels, centers):
    colors = ['b', 'y', 'm', 'g', 'c']
    fig, _ = plt.subplots(1, 1, figsize=(8, 5), dpi=120)
    fig.set_label('Kết quả')
    ax = plt.subplot(111, projection='3d')
    # img = ax.scatter(data.iloc[:,0], data.iloc[:,1], data.iloc[:,2], c=data.iloc[:,3],cmap=plt.hot())
    # c=fig.colorbar(img)
    # c.set_label('Chiều rộng đài hoa')
    mk = ['X', '^', 'D', '*', 'o']
    for i in range(len(centers)):
        color = np.random.choice(colors)
        colors.remove(color)
        type_ = data.iloc[np.where(labels == i)[0]]

        ax.scatter(type_.iloc[:, 0], type_.iloc[:, 1], type_.iloc[:, 2], c=color, s=type_.iloc[:, 3] * 50)
        ax.scatter(type_.iloc[:, 0], type_.iloc[:, 1], type_.iloc[:, 2], c='k', s=0.5)
        m_ = np.random.choice(mk)
        mk.remove(m_)
        s_ = ax.scatter([centers[i, 0]], [centers[i, 1]], [centers[i, 2]], color='r', marker=m_, s=100)
        ax.scatter([centers[i, 0]], [centers[i, 1]], [centers[i, 2]], color='k', marker=m_, s=0.5)
        s_.set_label('Trung tâm cụm ' + str(i + 1))
    note = ax.scatter([centers[1, 0]], [centers[1, 1]], [centers[1, 2]], color='w', s=0)
    note.set_label('Đường kính của mỗi điểm là chiều rộng của đài hoa')
    fig.legend(loc="upper right")
    ax.set_title('Biểu đồ dữ liệu sau khi chia')
    ax.set_xlabel('Chiều dài cánh hoa')
    ax.set_ylabel('Chiều rộng cánh hoa');
    ax.set_zlabel('Chiều dài đài hoa');
    fig.show()


# In[174]:


data=pd.read_csv('iris.csv')
kmeans = KMeans(n_clusters=3, random_state=0).fit(data.iloc[:, :4])
print("\n\n\n")
labels = np.array(kmeans.labels_)
centers = np.array(kmeans.cluster_centers_)
columns = np.array(data.columns)

showElbowOfDataToChooseCluster(data)

showCharts(data, labels, centers)

showBegin(data, labels, centers)

showFinal(data, labels, centers)

showCenters(centers, columns[:4])
