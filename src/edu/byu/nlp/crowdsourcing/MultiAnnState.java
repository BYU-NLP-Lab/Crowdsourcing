/**
 * Copyright 2013 Brigham Young University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.byu.nlp.crowdsourcing;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;

import com.google.common.base.Charsets;

import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.stats.DirichletDistribution;
import edu.byu.nlp.util.DoubleArrays;
import edu.byu.nlp.util.IntArrays;
import edu.byu.nlp.util.Matrices;

/**
 * Represents the state of all parameters in the MultiAnnModel
 */
public interface MultiAnnState {

  int[] getY();
  
  Map<String,Integer> getInstanceIndices();

  int[] getM();

  double[] getTheta();

  double[] getMeanTheta();

  double[][] getLogPhi();

  double[][] getMeanLogPhi();

  double[][][] getAlpha();
  
  double[][][] getMeanAlpha();

  double[][] getMu();

  double[][] getMeanMu();
  
  Dataset getData();
  
  int getNumAnnotators();
  
  int getNumLabels();

  void serializeToFile(String serializeToFile) throws IOException;

  void serializeTo(PrintWriter serializeToFile) throws IOException; 

  public void longDescription(PrintWriter serializeOut);

  public abstract class AbstractMultiAnnState implements MultiAnnState{
    @Override
    public int getNumAnnotators() {
      return getMeanAlpha().length;
    }
    @Override
    public int getNumLabels() {
      return getMeanTheta().length;
    }
    @Override
    public void serializeToFile(String serializeToFile) throws IOException {
      BufferedWriter writer = Files.newBufferedWriter(Paths.get(serializeToFile),Charsets.UTF_8);
      serializeTo(new PrintWriter(writer));
      writer.close();
    }
    @Override
    public void serializeTo(PrintWriter serializeOut){
      serializeOut.write(IntArrays.toString(getY()));
      serializeOut.write("\n");
      serializeOut.write(IntArrays.toString(getM()));
    }
    public void longDescription(PrintWriter serializeOut) {
      serializeOut.append("\n\n#########################\n");
      serializeOut.append("# Multiann Model Description\n");
      serializeOut.append("# num annotators: "+getNumAnnotators()+"\n");
      serializeOut.append("# num labels: "+getNumLabels()+"\n");
      serializeOut.append("#########################\n\n");
      serializeOut.append("y: "+IntArrays.toString(getY())+"\n\n");
      serializeOut.append("m: "+IntArrays.toString(getM())+"\n\n");
      serializeOut.append("theta: "+DoubleArrays.toString(getMeanTheta())+"\n\n");
      double[][][] alpha = getMeanAlpha();
      for (int a=0; a<getNumAnnotators(); a++){
        serializeOut.append("gamma"+a+": \n"+Matrices.toString(alpha[a])+"\n\n");
      }

      serializeOut.append("mu: \n"+Matrices.toString(getMeanMu())+"\n\n");
    }
  }

  public class BasicNeuteredMultiAnnState extends BasicMultiAnnState{
    public BasicNeuteredMultiAnnState(int[] y, double[] theta,
        double[] meanTheta, double[][] logPhi, double[][] meanLogPhi,
        double[][] meanMu, double[][][] alpha,
        double[][][] meanAlpha, Dataset data, Map<String,Integer> instanceIndices) {
      super(y, y.clone(), // set all m equal to y 
          theta, meanTheta, logPhi, meanLogPhi,
          // set mu and muMean equal to diagonal row matrices 
          Matrices.diagonalMatrix(theta.length, theta.length), Matrices.diagonalMatrix(theta.length, theta.length), 
//          Matrices.uniformRowMatrix(theta.length, theta.length), Matrices.uniformRowMatrix(theta.length, theta.length), // uniform row matrices 
          alpha, meanAlpha, data, instanceIndices);
    }
  }
  
  public class BasicMultiAnnState extends AbstractMultiAnnState{
    private final int[] y; 
    private final int[] m; 
    private final double[] theta, meanTheta;
    private final double[][] logPhi, meanLogPhi;
    private final double[][] mu, meanMu;
    private final double[][][] alpha, meanAlpha;
    private Dataset data;
    private Map<String,Integer> instanceIndices;
    public BasicMultiAnnState(int[] y, int[] m, double[] theta, double[] meanTheta,
        double[][] logPhi, double[][] meanLogPhi, double[][] mu, double[][] meanMu,
        double[][][] alpha, double[][][] meanAlpha, Dataset data, Map<String,Integer> instanceIndices){
      this.y=y;
      this.m=m;
      this.theta=theta;
      this.meanTheta=meanTheta;
      this.logPhi=logPhi;
      this.meanLogPhi=meanLogPhi;
      this.mu=mu;
      this.meanMu=meanMu;
      this.alpha=alpha;
      this.meanAlpha=meanAlpha;
      this.data=data;
      this.instanceIndices=instanceIndices;
    }
    @Override
    public int[] getY() {
      return y;
    }
    @Override
    public int[] getM() {
      return m;
    }
    @Override
    public double[] getTheta() {
      return theta;
    }
    @Override
    public double[] getMeanTheta() {
      return meanTheta;
    }
    @Override
    public double[][] getMu() {
      return mu;
    }
    @Override
    public double[][] getMeanMu() {
      return meanMu;
    }
    @Override
    public double[][][] getAlpha() {
      return alpha;
    }
    @Override
    public double[][][] getMeanAlpha() {
      return meanAlpha;
    }
    @Override
    public double[][] getLogPhi() {
      return logPhi;
    }
    @Override
    public double[][] getMeanLogPhi() {
      return meanLogPhi;
    }
    @Override
    public Map<String,Integer> getInstanceIndices() {
      return instanceIndices;
    }
    @Override
    public Dataset getData() {
      return data;
    }
  }

  public class CollapsedItemResponseState extends CollapsedNeuteredMultiAnnState{
    public CollapsedItemResponseState(int[] y, double[] logCountOfY,
        double[][][] countOfJYAndA, Dataset data,
        Map<String, Integer> instanceIndices, RandomGenerator rnd) {
      super(y, logCountOfY, 
          // countOfYAndX (phi) is a matrix of uniform stochastic row vectors (so as not to affect inference)
          Matrices.uniformRowMatrix(logCountOfY.length, data.getInfo().getNumFeatures()), 
          countOfJYAndA, data, instanceIndices, rnd);
    }
    
  }
  
  public class CollapsedNeuteredMultiAnnState extends CollapsedMultiAnnState{
    public CollapsedNeuteredMultiAnnState(int[] y,
        double[] logCountOfY, double[][] countOfYAndX,
        double[][][] countOfJYAndA,
        Dataset data, Map<String,Integer> instanceIndices,
        RandomGenerator rnd) {
      super(y, 
          // m are set equal to y
          y.clone(), logCountOfY, countOfYAndX,
//          // logCountOfMAndY (mu) is a matrix of uniform stochastic row vectors (so as not to affect inference)
//          Matrices.uniformRowMatrix(logCountOfY.length, logCountOfY.length), 
          // logCountOfMAndY (mu) is a matrix of diagonal stochastic row vectors (so as not to affect inference)
          Matrices.diagonalMatrix(logCountOfY.length, logCountOfY.length), 
          countOfJYAndA, data, instanceIndices, rnd);
    }
    @Override
    public double[][] getMu() {
      return logCountOfYAndM;
    }
    @Override
    public double[][] getMeanMu() {
      return logCountOfYAndM;
    }

  }
  
  /**
   * A sample from a collapsed model, which derives a sample
   * from sufficient statistics (but after drawing sample  
   * are once drawn, they are cached and permanent for this sample)
   */
  public class CollapsedMultiAnnState extends AbstractMultiAnnState{

    protected final int[] y; // inferred 'true' label assignments
    protected final int[] m; // features-only-ML label assignments
    protected final double[] logCountOfY; // (replaces theta)
    protected double[] theta;
    protected final double[][] countOfMAndX; // (replaces phi)
    protected double[][] phi;
    protected final double[][] logCountOfYAndM; // (replaces mu)
    protected double[][] mu;
    protected final double[][][] countOfJYAndA; // (replaces alpha)
    protected double[][][] alpha;
    private Dataset data;
    private Map<String,Integer> instanceIndices;

    protected final RandomGenerator rnd;

    public CollapsedMultiAnnState(int[] y, int[] m, double[] logCountOfY,
        double[][] countOfMAndX, double[][] logCountOfYAndM,
        double[][][] countOfJYAndA,  
        Dataset data, Map<String,Integer> instanceIndices,
        RandomGenerator rnd) {
      this.y = y;
      this.m = m;
      this.logCountOfY = logCountOfY;
      this.countOfMAndX = countOfMAndX;
      this.logCountOfYAndM = logCountOfYAndM;
      this.countOfJYAndA = countOfJYAndA;
      this.data=data;
      this.instanceIndices=instanceIndices;
      this.rnd = rnd;
    }
    @Override
    public int[] getY() {
      return y;
    }
    @Override
    public int[] getM() {
      return m;
    }
    /**
     * Returns a sample from theta based on its mean
     */
    @Override
    public double[] getTheta() {
      if (theta == null) {
        theta = DoubleArrays.exp(logCountOfY);
        DirichletDistribution.sampleToSelf(theta, rnd);
      }
      return theta;
    }
    @Override
    public double[] getMeanTheta() {
      double[] mean = DoubleArrays.exp(logCountOfY);
      DoubleArrays.normalizeToSelf(mean);
      return mean;
    }
    @Override
    public double[][] getLogPhi() {
      if (phi==null){
        phi = Matrices.clone(countOfMAndX);
        DirichletDistribution.logSampleToSelf(phi, rnd);
      }
      return phi;
    }
    @Override
    public double[][] getMeanLogPhi() {
      double[][] mean = Matrices.clone(countOfMAndX);
      Matrices.logToSelf(mean);
      Matrices.logNormalizeRowsToSelf(mean);
      return mean;
    }
    @Override
    public double[][][] getAlpha() {
      if (alpha==null){
        alpha = new double[countOfJYAndA.length][][];
        for (int j=0; j<getNumAnnotators(); j++){
          alpha[j] = DirichletDistribution.sample(countOfJYAndA[j], rnd);
        }
      }
      return alpha;
    }
    @Override
    public double[][][] getMeanAlpha() {
      int numAnnotators = countOfJYAndA.length;
      double[][][] means = new double[numAnnotators][][];
      for (int j=0; j<numAnnotators; j++){
        means[j] = Matrices.clone(countOfJYAndA[j]);
        Matrices.normalizeRowsToSelf(means[j]);
      }
      return means;
    }
    @Override
    public double[][] getMu() {
      if (mu==null){
        mu = Matrices.exp(logCountOfYAndM);
        DirichletDistribution.sampleToSelf(mu, rnd);
      }
      return mu;
    }
    @Override
    public double[][] getMeanMu() {
      double[][] mean = Matrices.exp(logCountOfYAndM);
      // normalize each dirichlet row vector to get the means
      Matrices.normalizeRowsToSelf(mean);
      return mean;
    }
    @Override
    public Map<String,Integer> getInstanceIndices() {
      return instanceIndices;
    }
    @Override
    public Dataset getData() {
      return data;
    }
    
  }


}
