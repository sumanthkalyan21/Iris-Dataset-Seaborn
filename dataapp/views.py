from django.shortcuts import render
# Create your views here.
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
@csrf_exempt
def index(request):
    # if post request came
    if request.method == 'POST':
        # getting values from post
        sepal_length = request.POST.get('length')
        petal_length = request.POST.get('length1')
        iris_data = load_iris()
        data = pd.DataFrame({'sepal_length': iris_data.data[:, 0],
                             'sepal_width': iris_data.data[:, 1],
                             'petal_length': iris_data.data[:, 2],
                             'petal_width': iris_data.data[:, 3],
                             'species': iris_data.target
                             })
        # print(data.head())
        X = data[['sepal_length', 'petal_length']]
        y = data['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        new_candidates = {'sepal_length': [2.5],
                          'petal_length': [3.5]
                          }
        new_candidates['sepal_length'][0] = float(sepal_length)
        new_candidates['petal_length'][0] = float(petal_length)
        df2 = pd.DataFrame(new_candidates, columns=['sepal_length', 'petal_length'])
        y_pred = clf.predict(df2)
        if (y_pred[0] == "Iris-setosa"):
            name="setosa"
        elif (y_pred[0] == "Iris-versicolor"):
            name= "versicolor"
        else:
            name= "virginica"
            #print(name)
        sns.scatterplot(
            x='sepal_length',
            y='petal_length',
            hue='species',
            data=X_test.join(y_test)
        )
        plt.savefig('C:/Users/sumanth/PycharmProjects/datapro/dataapp/static/my_app/scatter.png')
              # adding the values in a context variable
        context = {
            'name': name,
                    }

        # getting our showdata template
        template = loader.get_template('showdata.html')

        # returing the template
        return HttpResponse(template.render(context, request))
    else:
        # if post request is not true
        # returing the form template
        template = loader.get_template('index.html')
        return HttpResponse(template.render())