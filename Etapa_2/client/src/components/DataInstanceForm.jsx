import {
	addToast,
	Button,
	Card,
	CardHeader,
	DatePicker,
	Divider,
	Input,
	Switch,
	Textarea,
} from '@heroui/react';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { CheckCircle, CloseCircle, FileSmile } from 'solar-icon-set';
import UploadedData from './UploadedData';

export default function DataInstanceForm({ isRetrain }) {
	const { register, handleSubmit, watch, setValue, reset } = useForm();

	const [isLoading, setIsLoading] = useState(false);
	const [dataInstances, setDataInstances] = useState([]);

	const onSubmit = data => {
		console.log(data);
		data.Fecha = String(data.Fecha);
		if (isRetrain) {
			data.Label = data.Label ? 1 : 0;
		}
		setDataInstances(prev => [...prev, data]);
		reset(); // Limpia el formulario
	};

	const predict = () => {
		setIsLoading(true);

		fetch('http://127.0.0.1:8000/predict', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				instances: dataInstances,
			}),
		}).then((res, rej) => {
			if (res.ok) {
				res.json().then(data => {
					setIsLoading(false);

					const updatedInstances = dataInstances.map(
						(item, index) => ({
							...item,
							prediction: data[index].label,
							accuracy: Math.max(
								data[index].prob_class_0,
								data[index].prob_class_1
							),
						})
					);
					setDataInstances(updatedInstances);
				});
			}
			if (rej) {
				console.error(rej);
			}
		});
	};

	const retrain = () => {
		setIsLoading(true);

		fetch('http://127.0.0.1:8000/retrain', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
			},
			body: JSON.stringify({
				instances: dataInstances,
			}),
		}).then((res, rej) => {
			if (res.ok) {
				res.json().then(data => {
					setIsLoading(false);
					console.log(data);
					addToast({
						title: data.message,
						classNames: {
							base: 'flex flex-col gap-2',
						},
						endContent: (
							<span className='text-sm text-success-300 flex gap-2'>
								<span>
									Precisión: {data.precision.toFixed(3)}
								</span>
								<span>Recall: {data.recall.toFixed(3)}</span>
								<span>F1: {data.f1_score.toFixed(3)}`</span>
							</span>
						),
						timeout: 5000,
						variant: 'flat',
						color: 'success',
						radius: 'md',
					});
				});
			}
		});
	};

	return (
		<div className='flex items-start justify-start w-full gap-4'>
			<Card className='p-4 min-w-[350px]'>
				<CardHeader className='mb-4 flex flex-col gap-2 items-start'>
					<h2 className='text-xl font-bold'>Agregar Noticia</h2>
					<p className='text-sm'>
						Agrega una nueva noticia para{' '}
						{isRetrain ? 'reentrenar' : 'predecir'}.
					</p>
				</CardHeader>
				<form
					onSubmit={handleSubmit(onSubmit)}
					className='flex flex-col gap-4'
				>
					<Input
						size='sm'
						key={'ID'}
						type='text'
						label='ID'
						{...register('ID', { required: true })}
					/>
					<Input
						size='sm'
						key={'Titulo'}
						type='text'
						label='Título'
						{...register('Titulo', { required: true })}
					/>
					<Textarea
						isClearable
						label='Descrición'
						{...register('Descripcion', { required: true })}
					/>
					<DatePicker
						label='Fecha'
						value={watch('Fecha')}
						onChange={date => setValue('Fecha', date)}
					/>
					{isRetrain ? (
						<div className='flex flex-col  gap-2'>
							<label>Tipo de noticia</label>
							<Switch
								defaultChecked={false}
								checked={watch('Label', false)}
								onChange={value => {
									setValue('Label', !watch('Label'));
									console.log(watch('Label'));
								}}
								thumbIcon={({ isSelected }) =>
									isSelected ? (
										<CheckCircle
											className='text-success'
											color='success'
										/>
									) : (
										<CloseCircle
											className='text-danger'
											color='danger'
										/>
									)
								}
								color='success'
							/>
							<p className='text-small text-default-500'>
								Noticia
							</p>
						</div>
					) : (
						' '
					)}
					<Button
						type='submit'
						color='success'
						variant='shadow'
						className='text-success-800 font-semibold'
					>
						Agregar
					</Button>
				</form>
				<Divider className='my-4' />
				<h2 className='text-xl font-bold'>Carga un archivo JSON</h2>
				<p className='text-sm'>
					Carga un archivo JSON para agregar varias noticias.
				</p>
				<div className='cursor-pointer'>
				<Input
					type='file'
					accept='.json'
					onChange={e => {
						const file = e.target.files[0];
						const reader = new FileReader();
						reader.onload = event => {
							const jsonData = JSON.parse(event.target.result);
							setDataInstances(prev => [...prev, ...jsonData]);
						};
						reader.readAsText(file);
					}}
					color='primary'
					className='mt-2 gap-2 flex items-center'
					id='file-upload'
					endContent={<FileSmile size={30} iconStyle='Outline' />}
				/>
				</div>
			</Card>

			<UploadedData
				dataInstances={dataInstances}
				actionTitle={isRetrain ? 'Reentrenar' : 'Predecir'}
				action={isRetrain ? retrain : predict}
				isLoading={isLoading}
			/>
		</div>
	);
}
