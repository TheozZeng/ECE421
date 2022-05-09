syms s;
O1 = exp(s) / (1 + exp(s));
diff_O1 = diff(O1,s);

X = [ 1 -1;
      1 0 ;
      1 1 ];
y = [0;
     2;
     0];
w = inv(X.'*X)*X.'*y
y_bar = X*w

w = [2;0;-2]
x4 = [1;0;0];
x5 = [1;2;4];
y4 = w.'*x4
y5 = w.'*x5
