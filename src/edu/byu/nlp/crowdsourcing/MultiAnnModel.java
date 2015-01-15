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

import java.util.Map;

import edu.byu.nlp.util.IntArrayCounter;
import edu.byu.nlp.util.MatrixAverager;

/**
 * @author pfelt
 */
public interface MultiAnnModel {

  void setTemp(double temp);

  void sample();

  void sampleY();

  void sampleM();
  
  MultiAnnState getCurrentState();
  
  void maximize();
  
  void maximizeY();

  void maximizeM();
  
  Map<String,Integer> getInstanceIndices();

  IntArrayCounter getMarginalYs();

  IntArrayCounter getMarginalMs();

  MatrixAverager getMarginalYMs();
  
  double getDocumentWeight(int docIndex);

  double logJoint();
  
}
