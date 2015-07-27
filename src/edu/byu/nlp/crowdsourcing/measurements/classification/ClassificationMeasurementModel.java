package edu.byu.nlp.crowdsourcing.measurements.classification;

import java.io.PrintWriter;
import java.util.Map;

import com.google.common.base.Preconditions;
import com.google.gson.Gson;

import edu.byu.nlp.classify.util.ModelTraining.SupportsTrainingOperations;
import edu.byu.nlp.crowdsourcing.PriorSpecification;
import edu.byu.nlp.crowdsourcing.measurements.AbstractMeasurementModelBuilder.StaticMeasurementModelCounts;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;

public interface ClassificationMeasurementModel extends SupportsTrainingOperations{

  State getCurrentState();
  
  Map<String,Integer> getInstanceIndices();

  double lowerBound(ClassificationMeasurementModelCounts counts);

  double[] fitOutOfCorpusInstance(DatasetInstance instance);
  
  
  
  /**
   * Although state values represent variational parameters, 
   * I call them the same thing as the variables whose 
   * distributions they parameterize. This helps avoid 
   * the alphabet soup afflicting MeanFieldMultiAnnState. 
   *
   * @author plf1
   */
  public class State {
    
    private final PriorSpecification priors;
    private final Map<String,Integer> instanceIndices;
    private final double[] nuTheta; // C (one per class)
    private final double[][] nuSigma2; // J x 2 (each annotator has an alpha,beta parameters to an inverse gamma) 
    private final double[][] logNuY; // N X C (one per instance per class)
    private final Dataset data;
    private final StaticMeasurementModelCounts staticCounts;
    
    private State(PriorSpecification priors, Map<String,Integer> instanceIndices, Dataset data, StaticMeasurementModelCounts staticCounts,
        double[] nuTheta, double[][] nuSigma2, double[][] logNuY){
      this.priors=priors;
      this.instanceIndices=instanceIndices;
      this.data=data;
      this.staticCounts=staticCounts;
      this.nuTheta=nuTheta;
      this.nuSigma2=nuSigma2;
      this.logNuY=logNuY;
    }
    public double[][] getLogNuY(){
      return logNuY;
    }
    public Map<String,Integer> getInstanceIndices(){
      return instanceIndices;
    }
    public double[] getNuTheta(){
      return nuTheta;
    }
    public double[][] getNuSigma2(){
      return nuSigma2;
    }
    public Dataset getData(){
      return data;
    }
    public StaticMeasurementModelCounts getStaticCounts(){
      return staticCounts;
    }
    public int getNumAnnotators(){
      return data.getInfo().getNumAnnotators();
    }
    public int getNumClasses(){
      return data.getInfo().getNumClasses();
    }
    public int getNumDocuments(){
      return data.getInfo().getNumDocuments();
    }
    public PriorSpecification getPriors(){
      return priors;
    }
    
    public static class Builder{

      private PriorSpecification priors;
      private Map<String,Integer> instanceIndices;
      private double[] nuTheta; // C (one per class)
      private double[][] nuSigma2; // J x 2 (each annotator has an alpha,beta parameters to an inverse gamma) 
      private double[][] logNuY; // N X C (one per instance per class)
      private Dataset data;
      private StaticMeasurementModelCounts staticCounts;
      
      public Builder setLogNuY(double[][] logNuY){
        this.logNuY=logNuY;
        return this;
      }
      
      public Builder setPriors(PriorSpecification priors){
        this.priors=priors;
        return this;
      }
  
      public Builder setInstanceIndices(Map<String,Integer> instanceIndices){
        this.instanceIndices=instanceIndices;
        return this;
      }
  
      public Builder setNuTheta(double[] theta){
        this.nuTheta=theta;
        return this;
      }
  
      public Builder setNuSigma2(double[][] sigma2){
        this.nuSigma2=sigma2;
        return this;
      }
  
      public Builder setData(Dataset data){
        this.data=data;
        return this;
      }
  
      public Builder setStaticCounts(StaticMeasurementModelCounts staticCounts) {
        this.staticCounts=staticCounts;
        return this;
      }
  
  
      public void longDescription(PrintWriter serializeOut){
        serializeOut.print(new Gson().toJson(this));
      }
      
      public State build(){
        Preconditions.checkNotNull(priors);
        Preconditions.checkNotNull(instanceIndices);
        Preconditions.checkNotNull(data);
        Preconditions.checkNotNull(staticCounts);
        Preconditions.checkNotNull(nuTheta);
        Preconditions.checkNotNull(nuSigma2);
        Preconditions.checkNotNull(logNuY);
        return new State(priors, instanceIndices, data, staticCounts, nuTheta, nuSigma2, logNuY);
      }
    }

    public State copy() {
      return new Builder()
      .setData(data)
      .setInstanceIndices(instanceIndices)
      .setPriors(priors)
      .setNuSigma2(nuSigma2)
      .setNuTheta(nuTheta)
      .setLogNuY(logNuY)
      .build()
      ;
    }
  }

}
