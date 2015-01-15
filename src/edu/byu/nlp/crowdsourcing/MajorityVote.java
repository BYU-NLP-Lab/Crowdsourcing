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

import java.util.List;

import org.apache.commons.math3.linear.SparseRealMatrix;
import org.apache.commons.math3.random.RandomGenerator;

import edu.byu.nlp.classify.data.AbstractLabelChooser;
import edu.byu.nlp.dataset.Datasets;
import edu.byu.nlp.math.SparseRealMatrices;

public class MajorityVote extends AbstractLabelChooser {
    
//    private final Function<DatasetInstance, Integer> labelGetter = new LabelGetter();
    private final RandomGenerator rnd;
    
    public MajorityVote(RandomGenerator rnd) {
        this.rnd = rnd;
    }

    /** {@inheritDoc} */
    @Override
    protected List<Integer> labelsForNonEmpty(SparseRealMatrix annotations) {
      return SparseRealMatrices.countColumns(annotations, Datasets.INT_CAST_THRESHOLD).argMaxList(-1, rnd);
    }

}