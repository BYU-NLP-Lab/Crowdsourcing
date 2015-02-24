package edu.byu.nlp.crowdsourcing;

import java.util.List;
import java.util.Map;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import edu.byu.nlp.classify.data.DatasetLabeler;
import edu.byu.nlp.classify.eval.BasicPrediction;
import edu.byu.nlp.classify.eval.Prediction;
import edu.byu.nlp.classify.eval.Predictions;
import edu.byu.nlp.crowdsourcing.SerializableCrowdsourcingState.SerializableCrowdsourcingDocumentState;
import edu.byu.nlp.data.types.Dataset;
import edu.byu.nlp.data.types.DatasetInstance;

public class SerializedLabelLabeler implements DatasetLabeler{

	private SerializableCrowdsourcingState state;
	public SerializedLabelLabeler(SerializableCrowdsourcingState state){
		Preconditions.checkNotNull(state);
		this.state=state;
	}
	
	@Override
	public Predictions label(Dataset trainingInstances, Dataset heldoutInstances) {

		Map<String, SerializableCrowdsourcingDocumentState> serializedLabels = state.getPerDocumentState();
		
		
		List<Prediction> labeledPredictions = Lists.newArrayList();
		List<Prediction> unlabeledPredictions = Lists.newArrayList();
		List<Prediction> heldoutPredictions = Lists.newArrayList();

		for (DatasetInstance inst: trainingInstances){
			Preconditions.checkArgument(serializedLabels.containsKey(inst.getInfo().getSource()),"serialized label is not available for instance "+inst.getInfo().getSource());
			if (inst.hasAnnotations()){
				labeledPredictions.add(new BasicPrediction(serializedLabels.get(inst.getInfo().getSource()).getY(), inst));
			}
			else{
				unlabeledPredictions.add(new BasicPrediction(serializedLabels.get(inst.getInfo().getSource()).getY(), inst));
			}
		}
		for (DatasetInstance inst: heldoutInstances){
			Preconditions.checkArgument(serializedLabels.containsKey(inst.getInfo().getSource()),"serialized label is not available for instance "+inst.getInfo().getSource());
			heldoutPredictions.add(new BasicPrediction(serializedLabels.get(inst.getInfo().getSource()).getY(), inst));
		}

		int numAnnotators = trainingInstances.getInfo().getNumAnnotators();
		int numClasses = trainingInstances.getInfo().getNumClasses();
		double[] annotatorAccuracies = new double[numAnnotators];
		double[][][] annotatorConfusionMatrices = new double[numAnnotators][numClasses][numClasses];
		double machineAccuracy = -1;
		double[][] machineConfusionMatrix = new double[numClasses][numClasses];
		double logJoint = -1;
		return new Predictions(labeledPredictions, unlabeledPredictions, heldoutPredictions, 
				annotatorAccuracies, annotatorConfusionMatrices, machineAccuracy, machineConfusionMatrix, logJoint);
		
	}
	
}
