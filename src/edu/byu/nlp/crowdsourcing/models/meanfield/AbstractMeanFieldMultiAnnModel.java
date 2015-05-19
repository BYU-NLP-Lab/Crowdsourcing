package edu.byu.nlp.crowdsourcing.models.meanfield;

import java.util.ArrayList;

import edu.byu.nlp.crowdsourcing.TrainableMultiAnnModel;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.data.types.SparseFeatureVector.EntryVisitor;
import edu.byu.nlp.util.Matrices;

public abstract class AbstractMeanFieldMultiAnnModel extends TrainableMultiAnnModel implements MeanFieldMultiAnnModel {

	protected void calculateCurrentAnnotatorConfusions(double[][][] confusions, int[][][] annotations, double[][] logg) {
		int numAnnotators = confusions.length;
		int numClasses = confusions[0].length;
		int numDocs = annotations.length;
		Matrices.multiplyToSelf(confusions, 0); // clear
		
		for (int i=0; i<numDocs; i++){
			for (int k=0; k<numClasses; k++){
				double confidenceThatKIsTruth = Math.exp(logg[i][k]);
				for (int j=0; j<numAnnotators; j++){
					for (int kprime=0; kprime<numClasses; kprime++){
						confusions[j][k][kprime] += annotations[i][j][kprime] * confidenceThatKIsTruth;
					}
				}
			}
		}
	}
	
  protected double[][] perClassVocab(final Dataset data, final ArrayList<DatasetInstance> instances, double[][] logg){
    final double[][] perClassVocab = new double[data.getInfo().getNumClasses()][data.getInfo().getNumFeatures()];
    final double[][] perDocumentClassAssignments = Matrices.exp(logg);
    
    for (int d=0; d<instances.size(); d++){
      DatasetInstance inst = instances.get(d);
      final double[] docAssignment = perDocumentClassAssignments[d]; 
      // add each word to each class (proportional to its assignment to that class)
      inst.asFeatureVector().visitSparseEntries(new EntryVisitor() {
        @Override
        public void visitEntry(int index, double value) {
          for (int c=0; c<data.getInfo().getNumClasses(); c++){
            perClassVocab[c][index] += value*docAssignment[c];
          }
        }
      });
      
    }
    return perClassVocab;
  }
	
}
