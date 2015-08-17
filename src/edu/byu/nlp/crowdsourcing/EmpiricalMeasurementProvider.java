/**
 * Copyright 2012 Brigham Young University
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

import java.util.List;
import java.util.Set;

import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;

import edu.byu.nlp.data.FlatInstance;
import edu.byu.nlp.data.types.Measurement;
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;

/**
 * Provides annotation as found in recorded data
 * 
 * @author pfelt
 *
 */
public class EmpiricalMeasurementProvider<D> implements LabelProvider<D, Measurement> {
	
	private EmpiricalAnnotations<D, Integer> annotations;
	private Set<FlatInstance<D, Integer>> usedAnnotations = Sets.newIdentityHashSet();
  private int annotatorId;
	
  public EmpiricalMeasurementProvider(int annotatorId, EmpiricalAnnotations<D, Integer> annotations) {
    this.annotatorId=annotatorId;
		this.annotations = annotations;
	}

	/** {@inheritDoc} */
	@Override
  public Measurement labelFor(String source, D datum) {
	  return nextAnnotation(annotations.getAnnotationsFor(source,datum));
	}

	/**
	 * Annotates based on historical data for the annotator set in the 
	 * constructor. Returns the first annotation for this instance 
	 * that has not been returned before.  
	 */
  private Measurement nextAnnotation(Multimap<Integer, FlatInstance<D, Integer>> instanceAnnotations) {
    // return next unused annotation for this annotator
    List<FlatInstance<D, Integer>> annotationList = Lists.newArrayList(instanceAnnotations.get(annotatorId));
    Datasets.sortAnnotationsInPlace(annotationList);
    
    for (FlatInstance<D, Integer> ann: annotationList){
      if (!usedAnnotations.contains(ann)){
        // return earliest annotation that hasn't already been used
        usedAnnotations.add(ann);
        return ann.getMeasurement();
      }
    }
    return null;
    
  }

}
