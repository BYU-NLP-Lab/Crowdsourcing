package measurements;

import java.io.PrintWriter;
import java.util.Map;

import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.util.IntArrayCounter;

public interface MeasurementModel extends SupportsTrainingOperations{

  State getCurrentState();
  
  Map<String,Integer> getInstanceIndices();

  IntArrayCounter getMarginalYs();

  double logJoint();
  
  
  public interface State {

    int[] getY();
    
    Map<String,Integer> getInstanceIndices();

    double[] getTheta();

    double[] getMeanTheta();

    double[][] getLogPhi();

    double[][] getMeanLogPhi();

    double[][][] getGamma();
    
    double[][][] getMeanGamma();

    Dataset getData();
    
    int getNumAnnotators();
    
    int getNumLabels();

    public void longDescription(PrintWriter serializeOut);
  }
  
  public interface Priors{
    
  }
  
}
