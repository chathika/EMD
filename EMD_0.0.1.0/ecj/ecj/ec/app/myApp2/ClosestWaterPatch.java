package ec.app.myApp2;
import ec.*;
import ec.gp.*;
import ec.util.*;

public class ClosestWaterPatch extends GPNode
{
	public String toString() { return "closest-water-patch"; }

	public int expectedChildren() { return 2; }

	public void eval(final EvolutionState state,
				 final int thread,
				 final GPData input,
				 final ADFStack stack,
				 final GPIndividual individual,
				 final Problem problem)
	{
		NetlogoData data1 = ((NetlogoData)(input));
		NetlogoData data2 = (NetlogoData)(data1.clone());
		
		children[0].eval(state, thread, data1, stack, individual, problem); // patchset 
		data1.netlogoString = data1.netlogoString.insert(0," min-one-of ");
		children[1].eval(state, thread, data2, stack, individual, problem); // potential farms
		data1.netlogoString.append(" with [ num-occupying-farms = 0 ] [ distance ");
		
		StringBuffer potentialFarm = new StringBuffer(data2.netlogoString);
		data1.netlogoString.append(potentialFarm);
		data1.netlogoString.append("]");
	}
}