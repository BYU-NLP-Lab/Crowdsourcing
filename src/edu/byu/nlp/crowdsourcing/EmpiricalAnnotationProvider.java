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
import edu.byu.nlp.data.util.EmpiricalAnnotations;
import edu.byu.nlp.dataset.Datasets;

/**
 * Provides annotation as found in recorded data
 * 
 * @author pfelt
 *
 */
public class EmpiricalAnnotationProvider<D, L> implements LabelProvider<D, L> {
	
	private EmpiricalAnnotations<D, L> annotations;
	private Set<FlatInstance<D, L>> usedAnnotations = Sets.newIdentityHashSet();
  private int annotatorId;
	
  public EmpiricalAnnotationProvider(int annotatorId, EmpiricalAnnotations<D, L> annotations) {
    this.annotatorId=annotatorId;
		this.annotations = annotations;
	}

	/** {@inheritDoc} */
	@Override
	  public L labelFor(int source, D datum) {
	  return nextAnnotation(annotations.getAnnotationsFor(source,datum));
	}

	/**
	 * Annotates based on historical data for the annotator set in the 
	 * constructor. Returns the first annotation for this instance 
	 * that has not been returned before.  
	 */
  private L nextAnnotation(Multimap<Integer, FlatInstance<D, L>> instanceAnnotations) {
    // return next unused annotation for this annotator
    List<FlatInstance<D, L>> annotationList = Lists.newArrayList(instanceAnnotations.get(annotatorId));
    Datasets.sortAnnotations(annotationList);
    
    for (FlatInstance<D, L> ann: annotationList){
      if (!usedAnnotations.contains(ann)){
        // return earliest annotation that hasn't already been used
        usedAnnotations.add(ann);
        return ann.getAnnotation();
      }
    }
    return null;
    
  }

}
