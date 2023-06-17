import pprint
import unittest

# Simple implementation of Bidirectional Associative Memory
# BAM specifically belongs to the hetero-associative memory type, meaning it can return patterns of
# different sizes when given a particular input pattern. This is in contrast to auto-associative memory,
# where the input and output patterns are of the same size.
class SimpleBAM(object):
    """
    Simple implementation of Bidirectional Associative Memory
    BAM specifically belongs to the hetero-associative memory type, meaning it can return patterns of
    different sizes when given a particular input pattern. This is in contrast to auto-associative memory,
    where the input and output patterns are of the same size.
    Most human learning is associative in nature (neurons that fire together wire together).
    """

    def __init__(self, data, threshold=0):
        self.associated_vectors = []
        self.threshold = threshold

        # Store associations in bipolar form to the array
        for item in data:
            self.associated_vectors.append(
                [self.to_bipolar_vector(item[0]),
                 self.to_bipolar_vector(item[1])]
            )

        self.input_vector_len = len(self.associated_vectors[0][1])
        self.output_vector_len = len(self.associated_vectors[0][0])

        # Create an empty BAM matrix
        self.bam_matrix = [[0 for _ in range(self.input_vector_len)] for _ in range(self.output_vector_len)]

        # Compute BAM matrix from associations
        self.create_bam()

    # Transform vector to bipolar form [-1, 1]
    def to_bipolar_vector(self, input_vector):
        return [-1 if element == 0 else 1 for element in input_vector]

    # Create the BAM matrix by multiplying the input vector by the transpose of the output vector
    def create_bam(self):
        for assoc_pair in self.associated_vectors:
            first_vector = assoc_pair[0]
            second_vector = assoc_pair[1]
            for idx, xi in enumerate(first_vector):
                for idy, yi in enumerate(second_vector):
                    self.bam_matrix[idx][idy] += xi * yi

    # Return the binary association vector for the input
    def associated_vector(self, input_vector):
        input_vector = self.multiply_matrix_vector(input_vector)
        return self.to_binary_vector(input_vector)

    # Calculate the raw output vector by multiplying the input vector with the BAM matrix
    def multiply_matrix_vector(self, input_vector):
        raw_output_vector = [0] * self.input_vector_len
        for x in range(self.input_vector_len):
            for y in range(self.output_vector_len):
                raw_output_vector[x] += input_vector[y] * self.bam_matrix[y][x]
        return raw_output_vector

    # Transform the vector into a binary vector containing elements [0, 1]
    def to_binary_vector(self, vector):
        return [0 if element < self.threshold else 1 for element in vector]

    def bam_matrix(self):
        return self.bam_matrix

class BAM_Network_Test_Cases(unittest.TestCase):

    def test_simple_pattern(self):

        data_pairs = [
            [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0]]
        ]

        bidirectional_associative_network = SimpleBAM(data_pairs)

        assert(bidirectional_associative_network.associated_vector([1, 0, 1, 0, 1, 0]) == [1, 1, 0, 0])
        assert(bidirectional_associative_network.associated_vector([1, 1, 1, 0, 0, 0]) == [1, 0, 1, 0])

    def test_simple_pattern2(self):

        data_pairs = [
            [[1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 0]],
            [[1, 1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0]],
            [[0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 1]]
        ]

        bidirectional_associative_network = SimpleBAM(data_pairs)

        assert (bidirectional_associative_network.associated_vector([1, 0, 1, 0, 1, 0, 0, 0]) == [0, 1, 0, 0])
        assert (bidirectional_associative_network.associated_vector([1, 1, 1, 0, 0, 1, 0, 1]) == [1, 0, 0, 0])
        assert (bidirectional_associative_network.associated_vector([0, 0, 1, 1, 1, 1, 1, 0]) == [0, 0, 0, 1])


    def test_complex_pattern(self):

        zero = [
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            0, 1, 1, 1, 0
        ]

        one = [
            0, 1, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0
        ]

        two = [
            1, 1, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 1, 0,
            0, 1, 1, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
        ]

        half_zero = [
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ]

        data_pairs = [
            [zero,[1, 0, 0]],
            [one, [0, 1, 0]],
            [two, [0, 0, 1]]
        ]

        bidirectional_associative_network = SimpleBAM(data_pairs)

        assert (bidirectional_associative_network.associated_vector(zero) == [1, 0, 0])
        assert (bidirectional_associative_network.associated_vector(one) == [0, 1, 0])
        assert (bidirectional_associative_network.associated_vector(two) == [0, 0, 1])
        assert (bidirectional_associative_network.associated_vector(half_zero) == [1, 0, 0])


    def test_simple_print(self):

        data_pairs = [
            [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 0, 0, 0], [1, 0, 1, 0]]
        ]

        bidirectional_associative_network = SimpleBAM(data_pairs)

        pp = pprint.PrettyPrinter(indent=4)

        print('BAM Matrix:')
        pp.pprint(bidirectional_associative_network.bam_matrix())

        print('[1, 0, 1, 0, 1, 0] ---> ' + str(
            bidirectional_associative_network.associated_vector([1, 0, 1, 0, 1, 0])))
        print('[1, 1, 1, 0, 0, 0] ---> ' + str(
            bidirectional_associative_network.associated_vector([1, 1, 1, 0, 0, 0])))
        print('[1, 0, 1, 0, 0, 1] ---> ' + str(
            bidirectional_associative_network.associated_vector([1, 0, 1, 0, 0, 1])))

if __name__ == "__main__":
    unittest.main()