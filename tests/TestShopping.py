import unittest
from sklearn.model_selection import train_test_split
from shopping import load_data, train_model, evaluate


class TestShopping(unittest.TestCase):

    def setUp(self):
        self.evidence, self.labels = load_data("../shopping.csv")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=0.4
        )

    def test_load_data_returns_correct_length(self):
        self.assertEqual(len(self.evidence), len(self.labels))

    def test_train_model_returns_correct_type(self):
        model = train_model(self.x_train, self.y_train)
        self.assertEqual(type(model).__name__, 'KNeighborsClassifier')

    def test_evaluate_returns_sensitivity_and_specificity(self):
        model = train_model(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        sensitivity, specificity = evaluate(self.y_test, predictions)
        self.assertIsInstance(sensitivity, float)
        self.assertIsInstance(specificity, float)

    def test_evaluate_returns_values_between_zero_and_one(self):
        model = train_model(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        sensitivity, specificity = evaluate(self.y_test, predictions)
        self.assertTrue(0 <= sensitivity <= 1)
        self.assertTrue(0 <= specificity <= 1)


if __name__ == '__main__':
    unittest.main()
