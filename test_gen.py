import numpy as np



x = 5
y = 1

a = (np.random.rand(x+2,y)*10).astype(np.int32).astype(np.float32) / 2
b = (np.random.rand(y,x+4)*10).astype(np.int32).astype(np.float32) / 2
result = np.matmul(a,b)

print "    #[test]"
print "    fn test_dense_matmul_3(){"
print "        let a = DenseMatrix{"
print "            row_major_data:vec![{}],".format(", ".join(map(str,a.flatten())))
print "            n_rows:{},".format(a.shape[0])
print "            n_cols:{}".format(a.shape[1])
print "        };"
print "        let b = DenseMatrix{"
print "            row_major_data:vec![{}],".format(", ".join(map(str,b.flatten())))
print "            n_rows:{},".format(b.shape[0])
print "            n_cols:{}".format(b.shape[1])
print "        };"
print "        let result = matmul(&a,&b);"
print "        assert_eq!(result.row_major_data,"
print "            vec![{}]);".format(", ".join(map(str,result.flatten())))
print "        assert_eq!(result.n_rows, {});".format(result.shape[0])
print "        assert_eq!(result.n_cols, {});".format(result.shape[1])
print "    }"