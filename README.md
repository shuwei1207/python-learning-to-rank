# python-learning-to-rank

藉由python的套件Xgboost來加以實現。


一、	資料集：
公司機密。


二、	程式說明：
Step1：封包處理
先引入我們需要的封包
 
Step2：資料處理
再來就是將資料讀入，以股票代碼為index。 
最後將data1與data2資料標示0~4，資料前處理就完成。
 
Step3：資料分割（X & y）
  
Step4：放入資料
直接使用DMatrix來分析資料。
 
Step5：設置參數
之後便可以設置參數，一般是調整eta也就是learning rate來提高準確率，並可搭配不同的函數，也就是objective，且我們是multiclass的分類，必須填入class的數目。
 
Step6：開始訓練
使用DMatrix直接丟入即可訓練，會產出test資料中預測出來的y（best_preds）的label，即可和真實的y_test比較，印出精準度以供參考。
 
Step７：Confusion Matrix
用此矩陣可以幫助我們了解best_preds的label和y_test的label是否有對應，因為有5個class所以此矩陣會有25個，對角線（＼）代表預測精準的個數，可以幫助我們了解訓練出來的結果。
 
 
三、	結果呈現（可調整的參數）：
指標一：精準度
 
指標二：Confusion Matrix
 
指標三：挑出個股
此程式的主要目的是挑出L2R中判定為績優股的個股，所以會印出label是0的個股代碼。並印出個股與漲跌幅，即可算出大略的平均漲跌幅。
 
 
四、	降維處理（PCA與tSNE）：
因為所運算的項目太多，因此我們放入降維程式來觀察是否增加精準度：
［方法一］：PCA
PCA需要調整參數（n_components），來表示所需要的維度，其最大值只能是min( len(row) , len(column) ) = min( 48, 2000)= 48
 
［方法二］：tSNE
tSNE為自動運算降維方式，只須調整random_state，但須計算較久。
