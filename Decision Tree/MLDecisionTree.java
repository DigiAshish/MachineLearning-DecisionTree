import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import java.util.StringTokenizer;

public class MLDecisionTree {
	private static int count = 0;
	public static void main(String[] args) 
	{
	    if (args.length != 6)
	       {
	         System.err.println("You must call MLDecisionTree as " + 
	   			 "<L value (int)> <K value (int)> <path-to-training-set> <path-to-validation-set> <path-to-test-set> <to-print(yes/no)>\n");
	         System.exit(1);
	       }   
		int lvalue = Integer.valueOf(args[0]);
		int kvalue = Integer.valueOf(args[1]);
		String trainingFileLocation = args[2];
		String validationFileLocation = args[3];
		String testFileLocation = args[4];
		Boolean toPrint = args[5].equalsIgnoreCase("YES") ? true : false;
		
		int[] rowsAndColumns = DataSheetRowsColumn(trainingFileLocation);
		int[][] dataSheetValues = new int[rowsAndColumns[1]][rowsAndColumns[0]];
		String[] attributeNames = new String[rowsAndColumns[0]];
		int[] isAttributeUsed = new int[rowsAndColumns[0]];	
		int[] rowIndex = new int[dataSheetValues.length];
		for (int s = 0; s < rowsAndColumns[0]; s++) 
			isAttributeUsed[s] = 0;
		for (int s = 0; s < dataSheetValues.length; s++) 
			rowIndex[s] = s;
		
		dataSheetValues = getDatasheetValues(trainingFileLocation, dataSheetValues, attributeNames);
		node root = BuildDecisionTree(null, dataSheetValues, isAttributeUsed, rowsAndColumns[0] - 1, rowIndex, null,"Entropy");
		node prunedRoot = postPruning(validationFileLocation, lvalue, kvalue, root, dataSheetValues, rowsAndColumns[0] - 1);
		node varianceroot = BuildDecisionTree(null, dataSheetValues, isAttributeUsed, rowsAndColumns[0] - 1, rowIndex, null,"Variance");
		node variancePrunedRoot = postPruning(validationFileLocation, lvalue, kvalue, varianceroot, dataSheetValues, rowsAndColumns[0] - 1);
		System.out.println("Entropy Decision Tree Accuracy 	      : " + Accuracy(testFileLocation, root)+" %");	
		System.out.println("Pruned Entropy Decision Tree Accuracy : " + Accuracy(testFileLocation, prunedRoot)+" %");
		System.out.println("Variance Decision Tree Accuracy       : " + Accuracy(testFileLocation, varianceroot)+" %");
		System.out.println("Pruned Variance Decision Tree Accuracy: " + Accuracy(testFileLocation, variancePrunedRoot)+" %");	
		System.out.println("\n");
		if (toPrint)
		{
				System.out.println("The Entropy Decision Tree");
				printTree(root, 0, attributeNames);	
				System.out.println("The Pruned Entropy Decision Tree:");
				printTree(prunedRoot, 0, attributeNames);
				System.out.println("The Variance Decision Tree:");
				printTree(varianceroot, 0, attributeNames);
				System.out.println("The Pruned Variance Decision Tree:");
				printTree(variancePrunedRoot, 0, attributeNames);
		}	
	}
		
	private static int[] DataSheetRowsColumn(String fileLocation) {
		String line = "";int count = 0;int tokenNumber = 0;
		int[] rowsAndColumns = new int[2];
		BufferedReader br = null;
		try {
		br = new BufferedReader(new FileReader(fileLocation));
			while ((line = br.readLine()) != null) 
			{
				if (count == 0) {
					String[] countColumn = line.split(",");
					rowsAndColumns[0] = countColumn.length;
				}
				count++;
			}
		br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		rowsAndColumns[1] = count;
		return rowsAndColumns;
	}
	
	private static int[][] getDatasheetValues(String filePath, int[][] dataSheetValuesLoc, String[] attributeNames) {
		BufferedReader br = null;
		String line = "";
		try {
			br = new BufferedReader(new FileReader(filePath));
			int i = 0;
			while ((line = br.readLine()) != null) 
			{
				String[] lineParameters = line.split(",");
				int j = 0;
				if (i == 0) {
					for (String lineParameter : lineParameters) {
						attributeNames[j++] = lineParameter;
					}
				}
				else {
					for (String lineParameter : lineParameters) {
						dataSheetValuesLoc[i][j++] = Integer.parseInt(lineParameter);
					}
				}
				i++;
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dataSheetValuesLoc;
		}
			
	public static node BuildDecisionTree(node root, int[][] dataSheetValues, int[] isAttributeUsed, int width, int[] rowIndex, node parent,String method) {
		boolean allOnes = true; boolean allZeros = true; boolean allAttributeDone = true;
			if (root == null) {
				root = new node();
				for (int i : isAttributeUsed)
						if (i == 0)
							allAttributeDone = false;
				if (rowIndex == null || rowIndex.length == 0 || width == 1 || allAttributeDone) {
					root.isZeroOrOne = BinaryClassification(root, dataSheetValues, width);
					root.isLeaf = true;
					return root;
				}
				for (int i : rowIndex) {
					if (dataSheetValues[i][width] == 0) {
						allOnes = false;
						break;
					}
				}
				for (int i : rowIndex) {
					if (dataSheetValues[i][width] == 1) {
						allZeros = false;
						break;
					}
				}				
				if (allOnes || allZeros) {
					if (allOnes)
						root.isZeroOrOne = 1;
					if (allZeros)
						root.isZeroOrOne = 0;
					root.isLeaf = true;
					return root;
				}
			}
			root = getBestAttribute(root, dataSheetValues, isAttributeUsed, width,rowIndex,method);
			root.parent = parent;
			if (root.rootRowIndex != -1)
				isAttributeUsed[root.rootRowIndex] = 1;
			int leftSideAttributeVisited[] = new int[isAttributeUsed.length];
			int rightSideAttributeVisited[] = new int[isAttributeUsed.length];
			for (int j = 0; j < isAttributeUsed.length; j++) {
				leftSideAttributeVisited[j] = isAttributeUsed[j];
				rightSideAttributeVisited[j] = isAttributeUsed[j];

			}
			root.left = BuildDecisionTree(root.left, dataSheetValues, leftSideAttributeVisited, width, root.leftIndices, root, method);
			root.right = BuildDecisionTree(root.right, dataSheetValues, rightSideAttributeVisited, width, root.rightIndices, root, method);
			return root;
		}
		
		private static node getBestAttribute(node root, int[][] dataSheetValues, int[] isAttributeUsed, int width,int[] rowIndex, String method) {
			int i = 0;int k = 0;
			double maxGain = 0;int maxIndex = -1;int maxLeftIndex[] = null;int maxRightIndex[] = null;
			for (; i < width; i++) {
				if (isAttributeUsed[i] == 0) {
					double negatives = 0; double positives = 0;
					double rightPositives = 0;double rightNegatives = 0, leftPositives = 0, leftNegatives = 0;
					double left = 0;double right = 0;
					int[] leftIndex = new int[dataSheetValues.length];
					int[] rightIndex = new int[dataSheetValues.length];
					double methodValue = 0;double methodLeftValue = 0;double methodRightValue = 0;
					double Gain = 0;
					for (k = 0; k < rowIndex.length; k++) {
						if (dataSheetValues[rowIndex[k]][width] == 1) 
							positives++;
						else
							negatives++;
						if (dataSheetValues[rowIndex[k]][i] == 1) {
							rightIndex[(int) right++] = rowIndex[k];
							if (dataSheetValues[rowIndex[k]][width] == 1) 
								rightPositives++;
						   else
								rightNegatives++;
						}else {
							leftIndex[(int) left++] = rowIndex[k];
							if (dataSheetValues[rowIndex[k]][width] == 1)
								leftPositives++;
							else 
								leftNegatives++;
						}
					}
					if(method=="Entropy")
					{
					methodValue = (-1 * (Math.log(positives / rowIndex.length) / Math.log(2)) * ((positives / rowIndex.length)))
									+ (-1 * (Math.log(negatives / rowIndex.length) / Math.log(2)) * (negatives / rowIndex.length));
					methodLeftValue = (-1 * (Math.log(leftPositives / (leftPositives + leftNegatives)) / Math.log(2))
									* (leftPositives / (leftPositives + leftNegatives)))
									+ (-1 * (Math.log(leftNegatives / (leftPositives + leftNegatives)) / Math.log(2))
									* (leftNegatives / (leftPositives + leftNegatives)));
					methodRightValue = (-1 * (Math.log(rightPositives / (rightPositives + rightNegatives)) / Math.log(2))
									* (rightPositives / (rightPositives + rightNegatives)))
									+ (-1 * (Math.log(rightNegatives / (rightPositives + rightNegatives)) / Math.log(2))
									* (rightNegatives / (rightPositives + rightNegatives)));
					}
					if(method=="Variance")
					{
						methodValue = ((positives / rowIndex.length)) * (negatives / rowIndex.length);
						methodLeftValue = (leftPositives / (leftPositives + leftNegatives))
								* (leftNegatives / (leftPositives + leftNegatives));
						methodRightValue = (rightPositives / (rightPositives + rightNegatives))
								* (rightNegatives / (rightPositives + rightNegatives));
					}
					if (Double.compare(Double.NaN, methodValue) == 0) {
						methodValue = 0;
					}
					if (Double.compare(Double.NaN, methodLeftValue) == 0) {
						methodLeftValue = 0;
					}
					if (Double.compare(Double.NaN, methodRightValue) == 0) {
						methodRightValue = 0;
					}
					Gain = methodValue
							- ((left / (left + right) * methodLeftValue) + (right / (left + right) * methodRightValue));
					if (Gain >= maxGain) {
						maxGain = Gain;
						maxIndex = i;
						int leftTempArray[] = new int[(int) left];
						for (int index = 0; index < left; index++) {
							leftTempArray[index] = leftIndex[index];
						}
						int rightTempArray[] = new int[(int) right];
						for (int index = 0; index < right; index++) {
							rightTempArray[index] = rightIndex[index];
						}
						maxLeftIndex = leftTempArray;
						maxRightIndex = rightTempArray;
					}
				}
			}
			root.rootRowIndex = maxIndex;
			root.leftIndices = maxLeftIndex;
			root.rightIndices = maxRightIndex;
			return root;
		}
		public static int BinaryClassification(node root, int[][] dataSheetValues, int width) {
			int ones = 0;
			int zeros = 0;
			if (root.parent == null) {
				int i = 0;
				for (i = 0; i < dataSheetValues.length; i++) {			
					if (dataSheetValues[i][width] == 1) 
						ones++;
					 else
						zeros++;
				}
			} else {
				for (int i : root.parent.leftIndices) {
					if (dataSheetValues[i][width] == 1) 
						ones++;
					 else
						zeros++;
				}
				for (int i : root.parent.rightIndices) {
					if (dataSheetValues[i][width] == 1) 
						ones++;
					 else
						zeros++;
				}
			}
			return zeros > ones ? 0 : 1;
		}
		
		private static void printTree(node root, int printValues, String[] attributeNames) {
			int backupPrintValues = printValues;
			if (root.isLeaf) {
				System.out.println(" " + root.isZeroOrOne);
				return;
			}
			for (int i = 0; i < backupPrintValues; i++) 
				System.out.print("|  ");
			if (root.left != null && root.left.isLeaf && root.rootRowIndex != -1)
				System.out.print(attributeNames[root.rootRowIndex] + "= 0 :");
			else if (root.rootRowIndex != -1)
				System.out.println(attributeNames[root.rootRowIndex] + "= 0 :");
			printValues++;
			printTree(root.left, printValues, attributeNames);
			for (int i = 0; i < backupPrintValues; i++)
				System.out.print("|  ");
			if (root.right != null && root.right.isLeaf && root.rootRowIndex != -1)
				System.out.print(attributeNames[root.rootRowIndex] + "= 1 :");
			else if (root.rootRowIndex != -1)
				System.out.println(attributeNames[root.rootRowIndex] + "= 1 :");
			printTree(root.right, printValues, attributeNames);
		}
		
		private static double Accuracy(String filePath, node root) {
			int[] validationSetRowsAndColumns = DataSheetRowsColumn(filePath);	
			int[][] validationSetValues = new int[validationSetRowsAndColumns[1]][validationSetRowsAndColumns[0]];
			String[] validationSetAttributeNames = new String[validationSetRowsAndColumns[0]];
			validationSetValues = getDatasheetValues(filePath, validationSetValues, validationSetAttributeNames);
			double count = 0;
			for (int i = 1; i < validationSetValues.length; i++) {
				count += AccuracyCheck(validationSetValues[i], root);
			}
			double accuracyValue = (count / validationSetValues.length)*100;
			accuracyValue= (int)Math.round(accuracyValue * 1000)/(double)1000;;
			return accuracyValue;
		}
		
		private static int AccuracyCheck(int[] setValues, node newRoot) {
			int index = newRoot.rootRowIndex;
			int accuracyCount = 0;
			node testingNode = newRoot;
			while (testingNode.isZeroOrOne == -1) 
			{
				if (setValues[index] == 1)
					testingNode = testingNode.right;
				else 
					testingNode = testingNode.left;
				if (testingNode.isZeroOrOne == 1 || testingNode.isZeroOrOne == 0) {
					if (setValues[setValues.length - 1] == testingNode.isZeroOrOne) {
						accuracyCount = 1;
						break;
					} else 
						break;
				}
				index = testingNode.rootRowIndex;
			}
			return accuracyCount;
		}
		public static node postPruning(String filePath, int L, int K, node root, int[][]dataSheetValues, int width) {
			int i = 0;
			node postPrunedTree = new node();
			postPrunedTree = root;
			double maxAccuracy = Accuracy(filePath, root);
			for (i = 0; i < L; i++) {
				node newRoot = NodeDuplication(root);
				Random randNum = new Random();
				int M = 1 + randNum.nextInt(K);
				for (int j = 1; j <= M; j++) {
					count = 0;
					if (CountNumOfLeaf(newRoot) == 0)
						break;
					node nodesArray[] = new node[CountNumOfLeaf(newRoot)];
					FillArray(newRoot, nodesArray);
					int s = randNum.nextInt(CountNumOfLeaf(newRoot));
					nodesArray[s].isLeaf = true;
					nodesArray[s].isZeroOrOne = BinaryClassification(nodesArray[s],dataSheetValues, width);
					nodesArray[s].left = null;
					nodesArray[s].right = null;

				}
				double accuracy = Accuracy(filePath, newRoot);
				if (accuracy > maxAccuracy) {
					postPrunedTree = newRoot;
					maxAccuracy = accuracy;
				}
			}
			return postPrunedTree;
		}
		private static void FillArray(node root, node[] Array) {
			if (root == null || root.isLeaf)
				return;
			Array[count++] = root;
			FillArray(root.left, Array);
			FillArray(root.right, Array);
		}
		private static int CountNumOfLeaf(node root) {
			if (root == null || root.isLeaf)
				return 0;
			else
				return (1 + CountNumOfLeaf(root.left) + CountNumOfLeaf(root.right));
		}
		public static node NodeDuplication(node root) {
			if (root == null)
				return root;
			node temp = new node();
			temp.parent = root.parent;
			temp.leftIndices = root.leftIndices;
			temp.rightIndices = root.rightIndices;
			temp.isZeroOrOne = root.isZeroOrOne;
			temp.isLeaf = root.isLeaf;
			temp.rootRowIndex = root.rootRowIndex;			
			temp.left = NodeDuplication(root.left);
			temp.right = NodeDuplication(root.right);
			return temp;
		}
	}
	class node {
		node parent; node left; node right;
		int leftIndices[]; int rightIndices[];
		boolean isLeaf = false;
		int rootRowIndex = -1;
		int isZeroOrOne = -1;
	}