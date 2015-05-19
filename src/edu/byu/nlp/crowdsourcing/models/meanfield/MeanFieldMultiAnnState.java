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
package edu.byu.nlp.crowdsourcing.models.meanfield;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Map;

import cc.mallet.classify.MaxEnt;
import edu.byu.nlp.crowdsourcing.MultiAnnState;
import edu.byu.nlp.data.types.Dataset;


/**
 * Represents the state of all parameters in the MultiAnnModel
 */
public class MeanFieldMultiAnnState implements MultiAnnState {

  private double[][] g;
  private double[][] h;
  private double[] pi;
  private double[][][] nu;
  private double[][] tau;
  private double[][] lambda;
  private Dataset data;
  private Map<String, Integer> instanceIndices;
  private MaxEnt maxent;

  public MeanFieldMultiAnnState(double[][] g, double[][] h, double[] pi, 
      double[][][] nu, double[][] tau, double[][] lambda, Dataset data, Map<String, Integer> instanceIndices){
    this(g, h, pi, nu, tau, lambda, null, data, instanceIndices);
  }
  public MeanFieldMultiAnnState(double[][] g, double[][] h, double[] pi, 
      double[][][] nu, double[][] tau, double[][] lambda, MaxEnt maxent, Dataset data, Map<String, Integer> instanceIndices){
    this.g=g;
    this.h=h;
    this.pi=pi;
    this.nu=nu;
    this.tau=tau;
    this.lambda=lambda;
    this.maxent=maxent;
    this.data=data;
    this.instanceIndices=instanceIndices;
  }
  public double[][] getG() {
    return g;
  }
  public double[][] getH() {
    return h;
  }
  public double[] getPi() {
    return pi;
  }
  public double[][][] getNu() {
    return nu;
  }
  public double[][] getTau() {
    return tau;
  }
  public double[][] getLambda() {
    return lambda;
  }
  public MaxEnt getMaxent(){
    return maxent;
  }
  @Override
  public Dataset getData() {
    return data;
  }
  @Override
  public int getNumAnnotators() {
    return nu.length;
  }
  @Override
  public int getNumLabels() {
    return data.getInfo().getNumClasses();
  }
  @Override
  public Map<String, Integer> getInstanceIndices() {
    return instanceIndices;
  }
  
  
  
  /** {@inheritDoc} */
  @Override
  public int[] getY() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public int[] getM() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[] getTheta() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[] getMeanTheta() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[][] getLogPhi() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[][] getMeanLogPhi() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[][][] getAlpha() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[][][] getMeanAlpha() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[][] getMu() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public double[][] getMeanMu() {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public void serializeToFile(String serializeToFile) throws IOException {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public void serializeTo(PrintWriter serializeToFile) throws IOException {
    throw new UnsupportedOperationException();
  }
  /** {@inheritDoc} */
  @Override
  public void longDescription(PrintWriter serializeOut) {
    throw new UnsupportedOperationException();
  }
    
}
