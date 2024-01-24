// ridge regression functions
var Ridge = function(X,y,lambda){
    var obj = this;
    obj.X = X;
    obj.y = y;
    obj.lambda = lambda;
    obj.beta = null;
    obj.train = function(){
        // ridge regression
        // X'X + lambda*I
        var XtX = numeric.dot(numeric.transpose(obj.X),obj.X);
        var lambdaI = numeric.mul(obj.lambda,numeric.identity(XtX.length));
        var XtX_lambdaI = numeric.add(XtX,lambdaI);
        // X'y
        var Xty = numeric.dot(numeric.transpose(obj.X),obj.y);
        // beta = (X'X + lambda*I)^-1 * X'y
        obj.beta = numeric.solve(XtX_lambdaI,Xty);
        return obj.beta;
    }
    obj.predict = function(X){
        var y = numeric.dot(X,obj.beta);
        return y;
    }
    obj.score = function(X,y){
        var y_pred = obj.predict(X);
        var y_mean = numeric.sum(y)/y.length;
        var SST = numeric.sum(numeric.pow(numeric.sub(y,y_mean),2));
        var SSR = numeric.sum(numeric.pow(numeric.sub(y,y_pred),2));
        var R2 = 1 - SSR/SST;
        return R2;
    }
}

//test 
var X = [[1,1],[1,2],[1,3]];
var y = [3,5,7];
var lambda = 10;
var ridge = new Ridge(X,y,lambda);
console.log(ridge.train());
console.log(ridge.predict(X));
console.log(ridge.score(X,y));
