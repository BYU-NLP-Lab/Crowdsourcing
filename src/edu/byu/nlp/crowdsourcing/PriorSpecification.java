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

import com.google.common.base.Preconditions;

public class PriorSpecification {

  public static final double HYPERPARAM_LEARNING_CONVERGENCE_THRESHOLD = 1e-4;
	
  private boolean inlineHyperparamTuning;
  
  private double bTheta;
  private final double bMu;
  private final double cMu;
  private double bGamma;
  private double cGamma;
  private double bPhi;
  private int numAnnotators;

  /**
   * bGamma is only array valued for legacy support. All values of bGamma must be the same. 
   */
  @Deprecated
  public PriorSpecification(double bTheta, double bMu, double cMu, double[] bGamma, double cGamma, double bPhi, boolean inlineHyperparamTuning) {
	  this(bTheta, bMu, cMu, bGamma[0], cGamma, bPhi, inlineHyperparamTuning, bGamma.length);
  }
  
  public PriorSpecification(double bTheta, double bMu, double cMu, double bGamma, double cGamma, double bPhi, boolean inlineHyperparamTuning, int numAnnotators) {
    Preconditions.checkArgument(bTheta > 0.0);
    Preconditions.checkArgument(bMu > 0.0);
    Preconditions.checkArgument(cMu > 0.0);
    Preconditions.checkArgument(bGamma > 0.0);
    Preconditions.checkArgument(cGamma > 0.0);
    Preconditions.checkArgument(bPhi > 0.0);

    this.bTheta = bTheta;
    this.bMu = bMu;
    this.cMu = cMu;
    this.bGamma = bGamma;
    this.cGamma = cGamma;
    this.bPhi = bPhi;
    this.numAnnotators=numAnnotators;
    this.inlineHyperparamTuning=inlineHyperparamTuning;
  }

  public double getBTheta() {
    return bTheta;
  }
  
  public void setBTheta(double bTheta){
    this.bTheta=bTheta;
  }

  public double getBMu() {
    return bMu;
  }

  public double getCMu() {
    return cMu;
  }

  public double getBGamma() {
	  return bGamma;
  }
  
  @Deprecated
  public double getBGamma(int annotator) {
    return getBGamma();
  }
  
  public void setBGamma(double val){
	  bGamma = val;
  }

  public double getCGamma() {
    return cGamma;
  }
  
  public void setCGamma(double val){
	  this.cGamma = val;
  }

  public double getBPhi() {
    return bPhi;
  }
  
  public void setBPhi(double bPhi){
    this.bPhi=bPhi;
  }

  public int getNumAnnotators() {
    return numAnnotators;
  }
  
  public boolean getInlineHyperparamTuning(){
    return inlineHyperparamTuning;
  }
  
}