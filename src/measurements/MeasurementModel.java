package measurements;

import java.io.PrintWriter;
import java.util.Map;

import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;
import edu.byu.nlp.util.IntArrayCounter;

public interface MeasurementModel extends SupportsTrainingOperations{

  State getCurrentState();
  
  Map<String,Integer> getInstanceIndices();

  IntArrayCounter getMarginalYs();

  double logJoint();

  double[] fitOutOfCorpusInstance(DatasetInstance instance);
  
  
  /**
   * Although these values represent variational parameters, 
   * I call them the same thing as the variables whose 
   * distributions they parameterize. This helps avoid 
   * the alphabet soup afflicting MeanFieldMultiAnnState. 
   *
   * @author plf1
   */
  public interface State {

    double[][] getY();
    
    Map<String,Integer> getInstanceIndices();

    double[] getTheta();

    double[] getMeanTheta();

    double[][] getLogPhi();

    double[][] getMeanLogPhi();

    double[][] getSigma2();
    
    double[][] getMeanSigma2();

    Dataset getData();
    
    int getNumAnnotators();
    
    int getNumLabels();

    public void longDescription(PrintWriter serializeOut);
  }

}
