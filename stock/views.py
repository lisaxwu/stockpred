from django.shortcuts import render
from django.http import HttpResponse
#from getdatayf import getdata_yf
from yahoo_finance import Share
import pandas as pd
import datetime as dt
import numpy as np
#from util import get_data, plot_data
import matplotlib.pyplot as plt
import json
import csv
from django.template import RequestContext

class KNNLearner(object):
	def __init__(self,k=2, verbose=False):
		self.k = k
		pass  # move along, these aren't the drones you're looking for

	def addEvidence(self, dataX, dataY):
		self.dataX = dataX
		self.dataY = dataY

	def query(self, points):
		pnum = points.shape[0]
		myshape=np.array(self.dataY.shape)
		myshape[0]=pnum
		predY = np.ones(myshape)

		for i in range(0, pnum):
			distan=((self.dataX-points[i,:])**2).sum(axis=1)
			ind = np.argsort(distan)
			indk = ind[0:self.k]
			predY[i]=(self.dataY[indk].sum(axis=0))/self.k

		return predY

def getdata_yf(symbol='GOOG',st = dt.datetime(2007,12,31),\
                et = dt.datetime(2009,12,31)):
	stock = Share(symbol)
	a = stock.get_historical(st.strftime("%Y-%m-%d"),et.strftime("%Y-%m-%d"))

	A = [float(x['Adj_Close']) for x in a]
	B = [pd.to_datetime(x['Date']).to_pydatetime() for x in a]

	df = pd.DataFrame(index = B,columns=[symbol],data = A)
    #print(df)
	df=df.iloc[::-1]
	return df

def getxy (sdtrain = dt.datetime(2007,12,31),\
                edtest=dt.datetime(2011, 12, 31), \
                syms = 'GOOG',days = 5, A = 26):

	alld = getdata_yf(symbol=syms, st=sdtrain, et=edtest)

	allfe = pd.DataFrame(index=alld.index)
	alleng = allfe.shape[0]
	allfe['bb'] = pd.Series(np.zeros(alleng), index=allfe.index)
	allfe['mom'] = pd.Series(np.zeros(alleng), index=allfe.index)
	allfe['vola'] = pd.Series(np.zeros(alleng), index=allfe.index)

	A = 26

	for i in range(A-1, alleng):
		allfe.ix[i,'mom'] = alld.ix[i, syms] / alld.ix[i-A+1, syms] - 1
		allfe.ix[i,'vola'] = (alld.ix[i-A+1:i+1, syms] / alld.ix[i-A+1, syms]).std()
		allfe.ix[i,'bb'] = (alld.ix[i, syms] - alld.ix[i-A+1:i+1, syms].mean()) / (3 * alld.ix[i-A+1:i+1, syms].std())

	ally = pd.DataFrame(index=alld.index)
	ally['5daydr'] = pd.Series(np.zeros(alleng), index=ally.index)

	for i in range(0,alleng-days):
		ally.ix[i,'5daydr'] = alld.ix[i+days,syms]/alld.ix[i,syms] - 1

	trainx = allfe.iloc[A-1:int(alleng/3)]
	trainy = ally.iloc[A-1:int(alleng/3)]

	testx = allfe
	return trainx,trainy,testx,alld



def predica (sdtrain = dt.datetime(2007,12,31),\
                edtest=dt.datetime(2011, 12, 31), \
                syms = 'GOOG',days = 5):

	trainx, trainy, testx, alld = getxy(sdtrain,edtest,syms,days)

	learner = KNNLearner(k=3, verbose=False)  # constructor
	learner.addEvidence(trainx.values, trainy.values)  # training step

	tmp = learner.query(testx.values)

	predy = pd.DataFrame(data=tmp,index=testx.index,columns=['5daydr'])

	allprep = pd.DataFrame(data=(alld.loc[predy.index].values)*((predy+1).values),index = predy.index,columns=[syms])

	md = allprep.index[-1]+pd.DateOffset(1)
	date_index = pd.date_range(md, periods=days, freq='D')
	increase = pd.DataFrame(index=date_index, columns=[syms])
    # increase.iloc[:] = 'NaN'
	alltmp = pd.concat([allprep, increase]).shift(days)
	alld = pd.concat([alld, increase])
	alld['pred'] = pd.Series(alltmp[syms])
	alld.fillna(value=0,inplace=True)
	return alld


def pptojsonarray(alld):
	a = [{"date": alld.index[i].strftime("%Y-%m-%d"), "price": alld.iloc[i,0],"prep": alld.iloc[i,1]} for i in range(0, alld.shape[0])]
	return a

# -------------------------------/////////////////////////

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def getdata(request):
	df = getdata_yf(symbol='GOOG',st = dt.datetime(2007,12,31),et = dt.datetime(2009,12,31))
	b=[{"date":df.index[i].strftime("%Y-%m-%d"),"val":df.ix[i,'GOOG']} for i in range(0,df.shape[0])]
    #c=json.dumps(b)
	return HttpResponse(json.dumps(b))

def oror(request):
	csvfile = open('/Users/chanteywu/Desktop/stockpred/stock/data/aapl.csv', 'r')
	fieldnames = ("Date","Volume","Close","Average")
	reader = csv.DictReader( csvfile, fieldnames)
	a = [row for row in reader]
	a = a[1:]
	return render(request, "sp.html", {"csvdata" : json.dumps(a)})


def aapl(request):
	p = predica (sdtrain = dt.datetime(2014,1,1),edtest=pd.datetime.today(), syms = 'AAPL',days = 10)
	a = pptojsonarray(p)
	return render(request, "d3stock.html", {"csvdata" : json.dumps(a)})

# def pred(request):
# 	return render(request, "d3stock.html", {"csvdata" : json.dumps(a)})

def ajax(request):
	# if request.POST.has_key('client_response'):
		
		r = request.POST['client_response']
		print(r)
		d = request.POST['client_response2']
		print(d)
		a = predica (sdtrain = dt.datetime(2014,1,1),edtest=pd.datetime.today(), syms = r,days = int(d))
		b = pptojsonarray(a)
		# response_dict = {}
		# response_dict.update({'server_response': b })
		return HttpResponse(b)
	# else:
	# 	return render_to_response('d3stock.html', context_instance=RequestContext(request))

#json.dumps(df),content_type="application/json"