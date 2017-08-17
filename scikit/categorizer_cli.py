# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from lib import model_importer
model = model_importer.model()

title = input('Enter a title: ')
description = input('Enter a description: ')

df = pandas.DataFrame.from_dict([{'title': title, 'description': description}])
print(df)

print(model.predict(df[['title', 'description']]))
