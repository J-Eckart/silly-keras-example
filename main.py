#! /usr/bin/env python3
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from enum import Enum
import random
from typing import List
import re

MAX_INPUT = 100

class Operator(Enum):
    plus = 1
    minus = 0

def coerce_input_value(val: int) -> float:
    if val < 0 or val > MAX_INPUT:
        raise ValueError(f"Input must be between 0 and {MAX_INPUT}")
    return val / float(MAX_INPUT)

def coerce_output_value(val: int) -> float:
    if val < -MAX_INPUT or val > 2 * MAX_INPUT:
        raise ValueError(f"Output value must be between {-MAX_INPUT} and {2 * MAX_INPUT}")
    return (val + MAX_INPUT) / float(3 * MAX_INPUT)

def uncoerce_input_value(val: float) -> int:
    return round(val*float(MAX_INPUT))

def uncoerce_output_value(val: float) -> int:
    return round(val * 3 * MAX_INPUT - MAX_INPUT)

def coerce_operator(operator: Operator) -> float:
    return float(operator.value)

def uncoerce_operator(value: float) -> Operator:
    return Operator(int(value))

def generate_input_array(first_operand: int, second_operand: int, operator: Operator) -> List[float]:
    return [coerce_input_value(first_operand), coerce_input_value(second_operand), coerce_operator(operator)]

def input_array_from_expression(expression: str) -> List[float]:
    regex_str = r"(\d+)\s*([+-])\s*(\d+)"
    regex = re.compile(regex_str)
    try:
        m = regex.match(expression)
        if m.groups()[1] == "+":
            return generate_input_array(int(m.groups()[0]), int(m.groups()[2]), Operator.plus)
        else:
            return generate_input_array(int(m.groups()[0]), int(m.groups()[2]), Operator.minus)
    except:
        raise ValueError("Invalid operation")

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
        to_return.append(generate_input_array(random.randint(0, MAX_INPUT), random.randint(0, MAX_INPUT), op))
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
    print(f"Remaining raw error is {eval_res}, which correlates to an error in the actual result of approx. {eval_res * 3 * MAX_INPUT}")
    while True:
        try:
            expression = input(f"Please enter an operation in the format 'a + b' or 'a - b', where a and b are numbers between 0 and {MAX_INPUT}: ")
            res = model.predict([input_array_from_expression(expression)])
            print(f"Result of {expression} is (maybe):", uncoerce_output_value(res[0][0]))
        except ValueError:
            print("That doesn't look like elementary school level maths to me...")


if __name__ == "__main__":
    main()