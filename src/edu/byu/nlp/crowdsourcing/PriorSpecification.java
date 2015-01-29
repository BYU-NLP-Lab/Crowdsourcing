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
  private double bTheta;
  private final double bMu;
  private final double cMu;
  private final double[] bAlpha;
  private final double cAlpha;
  private double bPhi;

  public PriorSpecification(double bTheta, double bMu, double cMu, double[] bAlpha, double cAlpha, double bPhi) {
    Preconditions.checkArgument(bTheta > 0.0);
    Preconditions.checkArgument(bMu > 0.0);
    Preconditions.checkArgument(cMu > 0.0);
    Preconditions.checkNotNull(bAlpha);
    for (double b : bAlpha) {
      Preconditions.checkArgument(b > 0.0);
    }
    Preconditions.checkArgument(cAlpha > 0.0);
    Preconditions.checkArgument(bPhi > 0.0);

    this.bTheta = bTheta;
    this.bMu = bMu;
    this.cMu = cMu;
    this.bAlpha = bAlpha;
    this.cAlpha = cAlpha;
    this.bPhi = bPhi;
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

  public double getBGamma(int annotator) {
    return bAlpha[annotator];
  }

  public double getCGamma() {
    return cAlpha;
  }

  public double getBPhi() {
    return bPhi;
  }
  
  public void setBPhi(double bPhi){
    this.bPhi=bPhi;
  }

  public int getNumAnnotators() {
    return bAlpha.length;
  }
}