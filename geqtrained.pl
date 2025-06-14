:- dynamic neuron/4.

neuron((1, 1), geq-0, [(2, 1)-0.24219999999999997, (2, 2)- -0.05059999999999998, (2, 3)-0.9267999999999998], 1.0).
neuron((1, 2), geq-0, [(2, 1)- -0.5650000000000001, (2, 2)-0.6493000000000001, (2, 3)-0.35230000000000006], 1.0).
neuron((1, 3), geq-0, [(2, 1)-0.9635, (2, 2)- -0.1564, (2, 3)-0.44210000000000005], 1.0).
neuron((1, 4), geq-0, [(2, 1)- -0.1573, (2, 2)-0.7073, (2, 3)- -0.39520000000000005], 1.0).
neuron((1, 5), geq-0, [(2, 1)- -2.2847, (2, 2)- -0.5923999999999999, (2, 3)-1.3089], 1.0).
neuron((1, 6), geq-0, [(2, 1)-0.0642, (2, 2)-0.5277, (2, 3)-0.7879999999999999], 1.0).
neuron((1, 7), geq-0, [(2, 1)-0.6239, (2, 2)- -0.2904, (2, 3)-0.1563], 1.0).
neuron((1, 8), geq-0, [(2, 1)- -0.7166, (2, 2)- -0.5927000000000001, (2, 3)-0.2415], 1.0).
neuron((1, 9), geq-0, [(2, 1)-0.3636, (2, 2)- -0.8884, (2, 3)-0.6200999999999999], 1.0).
neuron((1, 10), geq-0, [(2, 1)- -0.0521, (2, 2)-0.81, (2, 3)- -0.5867], 1.0).
neuron((1, 11), geq-0, [(2, 1)- -0.4148, (2, 2)-0.735, (2, 3)- -0.5140999999999999], 1.0).
neuron((1, 12), geq-0, [(2, 1)-0.1819, (2, 2)- -0.9600000000000001, (2, 3)-0.0724], 1.0).
neuron((1, 13), geq-0, [(2, 1)- -9.1649, (2, 2)- -8.648100000000003, (2, 3)- -0.1946000000000021], 1.0).
neuron((2, 1), geq-0, [], 0.0).
neuron((2, 2), geq-0, [], 0.0).
neuron((2, 3), geq-0, [], 1.0).

:- dynamic input_buffer/3.

input_buffer((1, 1), [], 0).
input_buffer((1, 2), [], 0).
input_buffer((1, 3), [], 0).
input_buffer((1, 4), [], 0).
input_buffer((1, 5), [], 0).
input_buffer((1, 6), [], 0).
input_buffer((1, 7), [], 0).
input_buffer((1, 8), [], 0).
input_buffer((1, 9), [], 0).
input_buffer((1, 10), [], 0).
input_buffer((1, 11), [], 0).
input_buffer((1, 12), [], 0).
input_buffer((1, 13), [], 0).
input_buffer((2, 1), [], 0).
input_buffer((2, 2), [], 0).
input_buffer((2, 3), [], 0).

:- dynamic best_accuracy/1.

best_accuracy(0.2676056338028169).

