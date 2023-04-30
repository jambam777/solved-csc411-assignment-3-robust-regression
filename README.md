Download Link: https://assignmentchef.com/product/solved-csc411-assignment-3-robust-regression
<br>
<strong> </strong>In this assignment we will be working with the Boston Housing dataset. This dataset contains 506 entries. Each entry consists of a house price and 13 features for houses within the Boston area. We suggest working in python and using the scikit-learn package to load the data. <strong>Starter Code. </strong>Starter code written in Python is provided for Question 2.

<ol>

 <li><strong> Robust Regression. </strong>One problem with linear regression using squared error loss is that it can be sensitive to outliers. Another loss function we could use is the <em>Huber loss</em>, parameterized by a hyperparameter <em>δ</em>:</li>

</ol>

if |<em>a</em>| ≤ <em>δ</em>

)      if |<em>a</em>| <em>&gt; δ</em>

<ul>

 <li>Sketch the Huber loss <em>L<sub>δ</sub></em>(<em>y,t</em>) and squared error loss for <em>t </em>= 0, either by hand or using a plotting library. Based on your sketch, why would you expect the Huber loss to be more robust to outliers?</li>

 <li><strong> </strong>Just as with linear regression, assume a linear model:</li>

</ul>

<em>y </em>= <strong>w</strong><sup>&gt;</sup><strong>x </strong>+ <em>b.</em>

Give formulas for the partial derivatives <em>∂L<sub>δ</sub>/∂</em><strong>w </strong>and <em>∂L<sub>δ</sub>/∂b</em>. (We recommend you find a formula for the derivative <em>H<sub>δ</sub></em><sup>0</sup>(<em>a</em>), and then give your answers in terms of <em>H<sub>δ</sub></em><sup>0</sup>(<em>y </em>− <em>t</em>).)

1

CSC411 Fall 2018                                                                                                                                                 Homework 3

<ul>

 <li><strong> </strong>Write Python code to perform (full batch mode) gradient descent on this model. Assume the training dataset is given as a design matrix X and target vector y. Initialize <strong>w </strong>and <em>b </em>to all zeros. Your code should be vectorized, i.e. you should not have a for loop over training examples or input dimensions. You may find the function where helpful.</li>

</ul>

Submit your code as q1.py.

<ol start="2">

 <li><strong> Locally Weighted Regression.</strong>

  <ul>

   <li><strong> </strong>Given {(<strong>x</strong><sup>(1)</sup><em>,y</em><sup>(1)</sup>)<em>,..,</em>(<strong>x</strong><sup>(<em>N</em>)</sup><em>,y</em><sup>(<em>N</em>)</sup>)} and positive weights <em>a</em><sup>(1)</sup><em>,…,a</em><sup>(<em>N</em>) </sup>show that the solution to the <em>weighted </em>least squares problem</li>

  </ul></li>

</ol>

<strong>w</strong><sup>∗ </sup>= argmin                                    (1)

is given by the formula

<strong>w</strong><sup>∗ </sup>= <strong>X</strong><em><sup>T</sup></em><strong>AX</strong><strong>Ay                                                            </strong>(2)

where <strong>X </strong>is the design matrix (defined in class) and <strong>A </strong>is a diagonal matrix where <strong>A</strong><em><sub>ii </sub></em>=

<em>a</em>(<em>i</em>)

It may help you to review Section 3.1 of the csc321 notes<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>.

<ul>

 <li>Locally reweighted least squares combines ideas from k-NN and linear regression. For each new test example <strong>x </strong>we compute distance-based weights for each training example <em><sup>i </sup></em>, computes <strong>w</strong><sup>∗ </sup>= argmin</li>

</ul>

and predicts ˆ<em>y </em>= <strong>x</strong><em><sup>T</sup></em><strong>w</strong><sup>∗</sup>. Complete the implementation of locally reweighted least

squares by providing the missing parts for q2.py.

Important things to notice while implementing: First, do not invert any matrix, use a linear solver (numpy.linalg.solve is one example). Second, notice that

but if we use <em>B </em>= max<em><sub>j </sub>A<sub>j </sub></em>it is much more numerically stable as

overflows/underflows easily. <em>This is handled automatically in the scipy package with the </em>scipy.misc.logsumexp <em>function</em><a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a><em>.</em>

<ul>

 <li>Randomly hold out 30% of the dataset as a validation set. Compute the average loss for different values of <em>τ </em>in the range [10,1000] on both the training set and the validation set. Plot the training and validation losses as a function of <em>τ </em>(using a log scale for <em>τ</em>).</li>

 <li><strong> </strong>How would you expect this algorithm to behave as <em>τ </em>→ ∞? When <em>τ </em>→ 0? Is this what actually happened?</li>

</ul>

2

<a href="#_ftnref1" name="_ftn1">[1]</a> <a href="http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">http://www.cs.toronto.edu/</a><a href="http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">~</a><a href="http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">delve/data/boston/bostonDetail.html</a>

<a href="#_ftnref2" name="_ftn2">[2]</a> <a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html">http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html</a>

<a href="#_ftnref3" name="_ftn3">[3]</a> <a href="http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L02%20Linear%20Regression.pdf">http://www.cs.toronto.edu/</a><a href="http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L02%20Linear%20Regression.pdf">~</a><a href="http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L02%20Linear%20Regression.pdf">rgrosse/courses/csc321_2018/readings/L02%20Linear%20Regression.pdf</a>

<a href="#_ftnref4" name="_ftn4">[4]</a> <a href="https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.misc.logsumexp.html">https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.misc.logsumexp.html</a>