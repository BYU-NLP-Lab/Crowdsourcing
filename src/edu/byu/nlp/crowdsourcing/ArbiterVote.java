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

import java.util.Collections;
import java.util.List;
import java.util.Set;

import org.apache.commons.math3.linear.SparseRealMatrix;
import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.math.AbstractRealMatrixPreservingVisitor;
import edu.byu.nlp.math.SparseRealMatrices;
import edu.byu.nlp.util.Counter;
import edu.byu.nlp.util.DenseCounter;
import edu.byu.nlp.util.Integers;
import edu.byu.nlp.util.Integers.MutableInteger;

/**
 * Selects the arbiter's labels; backs off to majority vote if there is none.
 */
public class ArbiterVote extends AbstractLabelChooser {
    private final Set<Long> arbiters;
    private final MajorityVote majorityVoteDelegate;
    
    public ArbiterVote(Set<Long> arbiters, RandomGenerator rnd) {
        this.arbiters = arbiters;
        this.majorityVoteDelegate = new MajorityVote(rnd);
    }
    
    @Override
    protected List<Integer> labelsForNonEmpty(SparseRealMatrix annotations) {
      
        int numAnnotations = (int) Math.round(SparseRealMatrices.sum(annotations));
      
        if (numAnnotations==0) {
            throw new IllegalStateException("Should have at least one annotation");
        }
        
        if (numAnnotations > 3) {
            throw new IllegalStateException("There should be at most three annotations: " + annotations);
        }

        // Search for the arbiter's annotation.
        final Counter<Integer> numArbiterAnnotations = new DenseCounter(1); // only has one entry
        final MutableInteger label = Integers.MutableInteger.from(null);
        
        annotations.walkInOptimizedOrder(new AbstractRealMatrixPreservingVisitor() {
          @Override
          public void visit(int annotator, int annval, double value) {
            if (arbiters.contains(annotator)) {
              numArbiterAnnotations.incrementCount(0, 1);
              if (numArbiterAnnotations.getCount(0) > 1) {
                  throw new IllegalStateException("Got more than one arbiter annotation");
              }
              label.setValue(annval);
            }
          }
        });
        if (label.getValue() != null) {
            return Collections.singletonList(label.getValue());
        }
        
        // There was no arbiter, so back off to majority vote
        return majorityVoteDelegate.labelsFor(annotations);
    }


}