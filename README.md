# LCY-MLTools #
----
Some supplementary procedures in Matlab for pattern recognition tools *[PRTools](http://37steps.com/prhtml/prtools.html)*  version 5.0.2 

This LCY-MLTools is maintained by LCY Machine Learning Research Couple.



##LCY-MLTools include the following procedures##
1.  Some procedures to extend the functionalities of  *[PRTools](http://37steps.com/prhtml/prtools.html))*


		* LC_ConfusionMatrix: calculate confusion matrix for given trained classifiers and test dataset.
 
2. Classification Algorithms

		* LC_MulticlassLogitBoost: LogitBoost for multiclass classification

##About the names of the routines##
1. **Starte with 'Data':** rontines related to datasets loading, converting or preprocessing.<br>
2. **Starte with 'Plot':** rontines related to figure plotting.<br>
3. **Starte with 'Eval':** rontines related to algorithm evaluation.<br>
4. **Starte with 'Wrap':** wrapper rontines, often related to LibSVM or Weka.
5. **Contain 'OCL':** rontines for one-class learning only.



##Tips##
- Datasets used in MLAT must be formatted in one the those three types:<br>


&emsp;	**PRTools5 style:** Save the prdataset data structure in the .mat file with name 'x'. <br>
&emsp;&emsp;	(For details of PRTools, please visit [www.37steps.com](www.37steps.com))<br>
&emsp;	**Weka(ARFF) style:** To be illustrated<br>
&emsp;	**LibSVM style:** To be illustrated<br>

- To run the MATLAB version of LCY-MLTools, Please download PRTools5 from [here](http://prtools.org) first. <br>


##Dependences (for MATLAB implementation)##
- PRTools is a pattern recognition toolbox in MATLAB maintained by [37steps](http://www.37steps.com). Note that PRTools is NOT an open source software, the copyright is hold by [37steps](http://www.37steps.com).<br>
- Weka is an open source data mining toolbox implemented in Java. LCY-MLTools utilized some functions of Weka via wrapper rontines.

##History##
 - 2013-11-27 Repo created