#! /usr/bin/env python3
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from enum import Enum
import random
from typing import List
import re

class Operator(Enum):
    plus = 1
    minus = 0

def coerce_input_value(val: int) -> float:
    """
    coerce an input number to be fed into the model into the range [0..1]
    val: a value in range [0, 10]
    returns: a value in range [0,1], where 0 corresponds to 0, 0.1 to, ... and 1 to 10
    """
    if val < 0 or val > 10:
        raise ValueError("Input must be between 0 and 10")
    return val / 10.0

def coerce_output_value(val: int) -> float:
    if val < -10 or val > 20:
        raise ValueError("Input must be between -10 and 30")
    return (val + 10) / 30.0

def uncoerce_input_value(val: float) -> int:
    return round(val*10.0)

def uncoerce_output_value(val: float) -> int:
    return round(val * 30.0 - 10.0)

def coerce_operator(operator: Operator) -> float:
    return float(operator.value)

def uncoerce_operator(value: float) -> Operator:
    return Operator(int(value))

def generate_input_array(first_operand: int, second_operand: int, operator: Operator) -> List[float]:
    return [coerce_input_value(first_operand), coerce_input_value(second_operand), coerce_operator(operator)]

def input_array_from_expression(expression: str) -> List[float]:
    regex_str = r"(\d)\s*([+-])\s*(\d)"
    regex = re.compile(regex_str)
 #   try:
    m = regex.match(expression)
    if m.groups()[1] == "+":
        return generate_input_array(int(m.groups()[0]), int(m.groups()[2]), Operator.plus)
    else:
        return generate_input_array(int(m.groups()[0]), int(m.groups()[2]), Operator.minus)
#    except:
#        raise ValueError("Invalid operation")

def create_model() -> keras.Sequential:
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(1, activation="relu")
    ])
    return model

def generate_training_input(size: int):
    to_return = []
    for i in range(size):
        op = Operator.plus
        if random.randint(0, 1) > 0:
            op = Operator.minus
        to_return.append(generate_input_array(random.randint(0, 10), random.randint(0, 10), op))
    return to_return

def generate_training_output(training_input: List):
    to_return = []
    for input in training_input:
        multiplier = 1.0
        if uncoerce_operator(input[-1]) == Operator.minus:
            multiplier = -1.0

        val = uncoerce_input_value(input[0]) + multiplier * uncoerce_input_value(input[1])
        to_return.append([coerce_output_value(val)])
    return to_return

def main():
    print("Running keras version", keras.__version__, "!")
    print("Creating model....")
    model = create_model()
    model.summary()
    print("Compiling model....")
    model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    training_size = int(input("Please choose number of operations to train on: "))
    x_train = generate_training_input(training_size)
    y_train = generate_training_output(x_train)
    x_test = generate_training_input(10)
    y_test = generate_training_output(x_test)
    print(f"Training on {training_size} operations....")
    model.fit(x_train, y_train, verbose=0)
    print("Evaluating....")
    eval_res = model.evaluate(x_test, y_test, verbose=0)
    print(f"Remaining raw error is {eval_res}, which correlates to an error in the actual result of approx. {eval_res * 30.0}")
    while True:
        """
        op_input = input("Please choose operator. + for plus, - for minus, anything else to exit: ")
        op = None
        if op_input == "+":
            op = Operator.plus
        elif op_input == "-":
            op = Operator.minus
        else:
            print("Bye!")
            exit(0)
        try:
            arg1 = int(input("Enter first number in range [0, 10]: "))
            arg2 = int(input("Second number: "))
            res = model.predict([generate_input_array(arg1, arg2, op)])
        except ValueError:
            print("Please make sure to input integers between 0 and 10!")
            continue
        """
        expression = input("Please enter an operation in the format 'a + b' or 'a - b', where a and b are numbers between 0 and 10: ")
        res = model.predict([input_array_from_expression(expression)])

        print(f"Result of {expression} is (maybe):", uncoerce_output_value(res[0][0]))

if __name__ == "__main__":
    main()