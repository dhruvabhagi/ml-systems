import numpy as np
from src.linearclass import util
import os

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path as a 1D numpy array
    log_reg = LogisticRegression() 
    log_reg.fit(x_train, y_train)
    y_predict = log_reg.predict(x_valid)
    np.savetxt(save_path, y_predict)

    util.plot(x_valid, y_valid, log_reg.get_theta(), os.path.join(script_dir, "validation_plot.png"), correction=1.0)

    log_reg._compute_training_loss(x_train, y_train)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def _compute_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_gradient(self,x, y, theta):
        h = self._compute_sigmoid(x @ theta)
        m, _ = x.shape
        return (1/m) * (x.T @ (h - y))

    def _compute_hessian(self, x, theta):
        h = self._compute_sigmoid(x @ theta)
        m, _ = x.shape
        h_product = h * (1-h)
        return (1/m) * (x.T @ (h_product[:, np.newaxis] * x))

    def _compute_training_loss(self, x, y):
        h = self._compute_sigmoid(x @ self.theta)
        loss = -np.mean((y * np.log(h + self.eps)) + ((1 - y) * np.log(1 - h + self.eps)))
        if self.verbose:
            print("computed loss on training={}".format(loss))
        return loss

    def get_theta(self):
        return self.theta
        
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        _, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        while True:
            gradient = self._compute_gradient(x, y, self.theta)
            hessian = self._compute_hessian(x, self.theta)
            try:
                theta_diff = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                print("Got error while solve(hessian, gradient), probably Hessian was singular?")
            if np.linalg.norm(theta_diff, 1) < self.eps:
                if (self.verbose):
                    print("breaking the while loop as the theta_diff={} is less than epsilon={}".format(theta_diff, self.eps))
                break
            self.theta = self.theta - theta_diff
        if self.verbose:
            print("computed theta={}".format(self.theta))
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self._compute_sigmoid(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    main(train_path=os.path.join(script_dir, 'ds1_train.csv'),
         valid_path=os.path.join(script_dir, 'ds1_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_1.txt'))

    main(train_path=os.path.join(script_dir, 'ds2_train.csv'),
         valid_path=os.path.join(script_dir, 'ds2_valid.csv'),
         save_path=os.path.join(script_dir, 'logreg_pred_2.txt'))