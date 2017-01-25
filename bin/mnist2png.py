# Sometimes it's convenient to visualize the data you're working with.
# This will consume an idx file and write out it's contents as pngs

import idx2numpy, numpy, png, sys

if len(sys.argv) != 3:
  print 'Usage: script.py <input file> <output directory>'
  sys.exit()

input_path = sys.argv[1]
output_path = sys.argv[2]

digits = idx2numpy.convert_from_file(input_path)

count, _, _ = digits.shape

for i in range(count):
  png.from_array(digits[i], 'L').save(output_path + '/digits_test_sample_%d.png' % i)
